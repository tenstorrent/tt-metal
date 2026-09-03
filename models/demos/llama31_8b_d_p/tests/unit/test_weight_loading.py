# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-WEIGHTS` — the REAL checkpoint loads with nothing missing and nothing silently unused,
and a cache-only rebuild is bit-identical.

Three assertions, each aimed at one specific silent failure (``BRINGUP_RECIPE.md:766-772``):

1. **No missing key.** A key this package looks for and the checkpoint does not have. Each one is a
   module that falls back to the weight cache — or, with no cache, fails at its first matmul with a
   message about an argument the caller never passed.
2. **No silently-unused key.** A key the checkpoint carries and nothing here reads. Benign in
   isolation; catastrophic when it is the *renamed twin* of a key that is simultaneously missing.
   That pair is precisely "a renamed key means a layer quietly runs on the wrong weights", and
   :func:`test_meta_renaming_is_caught_by_the_audit` shows the audit catching the exact rename the
   recipe suggested (``DEC-039``).
3. **Cache-only rebuild is bit-identical.** ``ttnn.as_tensor`` caches the *already-sharded*
   per-device tensor, so cache-only mode (``state_dict={}`` + a populated cache) is load-bearing for
   the P10 runner and is where a stale or wrong-shape cache turns into "one layer runs on garbage"
   (``R-017``). Compared by SHA-256 over every one of the 291 device tensors, model A freed before
   model B is built so only one copy is ever resident.

Two things this gate additionally proves, because the count alone would not:

* **the values are the checkpoint's**, not merely *some* 291 tensors —
  :func:`test_device_weights_match_the_checkpoint` reads a strided sample back off the device and
  compares against the checkpoint tensor put through the module's own layout and dtype ladder
  (transpose, Q/K Meta swizzle, ttnn quantisation). A layer running on the *previous* layer's
  weights passes every count-based check;
* **the mesh shape is in the cache path** (``R-017``), asserted directly on
  ``ModelArgs.weight_cache_path``.

Mesh: ``(1,1)``. P8 re-runs the cache-only assertion on ``(4,8)`` — one card cannot prove the
sharded cache (``Appendix F.10``).

Run:
    export HF_MODEL=/path/to/Llama-3.1-8B-Instruct
    pytest models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py -x -q
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama31_8b_d_p.tests.test_factory import (
    TestFactory,
    hf_model_path,
    llama_config_dims,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs, llama_hf_config, state_dict_uses_meta_keys
from models.demos.llama31_8b_d_p.utils.substate import substate

# 03_OUTLINE.md §4.1: 9 per-layer weights * 32 layers + embed_tokens + norm + lm_head.
EXPECTED_KEY_COUNT = 9 * 32 + 3

# Layers read back off the device and compared value-by-value against the checkpoint. Strided
# rather than exhaustive: the failure this catches (layer i built from layer j's weights, or from a
# stale cache) is a systematic indexing error, so the first, second, a middle and the last layer
# pin it. Reading all 32 would move ~9 GiB off the device for no extra discriminating power.
SAMPLED_LAYERS = (0, 1, 16, 31)


def _sha(t) -> str:
    """SHA-256 of a device tensor's host-side fp32 bytes.

    ``ttnn.to_torch`` of a ``bfloat8_b`` tensor returns the exact stored values widened to fp32, so
    equal hashes mean equal *stored* tensors — which is what "bit-identical device tensors" means
    for a gate that cannot read raw device memory.
    """
    a = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().contiguous()
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()


def _hash_all(model) -> dict:
    return {name: _sha(t) for name, t in model.named_device_tensors().items()}


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_no_missing_and_no_unused_keys(mesh_device, state_dict):
    """(a) of the gate: the checkpoint's key set and the model's consumed key set are equal.

    Both sets are printed in full on failure, as the recipe requires — the *identity* of a missing
    key is the whole diagnostic, and a bare count would hide a rename (which moves one key from
    ``consumed`` to ``unused`` and back, leaving the count at 291).
    """
    objs = TestFactory.setup_test(mesh_device)
    args = ModelArgs(mesh_device, weights_path=hf_model_path(), hf_config=objs["hf_config"])

    assert len(state_dict) == EXPECTED_KEY_COUNT, (
        f"the checkpoint has {len(state_dict)} tensors, expected {EXPECTED_KEY_COUNT} "
        f"(9*32 + 3, 03_OUTLINE.md §4.1)"
    )
    assert not state_dict_uses_meta_keys(state_dict), "the checkpoint is already Meta-keyed (DEC-039)"

    model = Model(
        mesh_device,
        objs["hf_config"],
        state_dict,
        mesh_config=objs["mesh_config"],
        ccl_manager=objs["ccl_manager"],
        max_seq_len=128,
        with_lm_head=True,
    )
    consumed = model.consumed_state_dict_keys()
    missing = consumed - set(state_dict)
    unused = set(state_dict) - consumed

    logger.info(
        f"[G-WEIGHTS] checkpoint keys = {len(state_dict)} | model consumed = {len(consumed)} | "
        f"missing = {len(missing)} | unused = {len(unused)}"
    )
    logger.info(f"[G-WEIGHTS] missing set: {sorted(missing)}")
    logger.info(f"[G-WEIGHTS] unused  set: {sorted(unused)}")

    # ModelArgs' expectation is derived from hf_config, independently of both sets above, so a
    # third-party disagreement shows up here rather than being defined away.
    args_missing, args_unused = args.audit_state_dict_keys(state_dict)
    assert (args_missing, args_unused) == (set(), set()), (
        f"ModelArgs.audit_state_dict_keys disagrees with the checkpoint: missing "
        f"{sorted(args_missing)}, unused {sorted(args_unused)}"
    )
    assert consumed == args.expected_state_dict_keys(), (
        "the model consumed a different key set than ModelArgs expects; one of the two is wrong "
        f"(symmetric difference: {sorted(consumed ^ args.expected_state_dict_keys())})"
    )

    assert not missing, f"[G-WEIGHTS] {len(missing)} checkpoint keys MISSING: {sorted(missing)}"
    assert not unused, f"[G-WEIGHTS] {len(unused)} checkpoint keys SILENTLY UNUSED: {sorted(unused)}"
    assert len(model.named_device_tensors()) == EXPECTED_KEY_COUNT, (
        f"the model holds {len(model.named_device_tensors())} device weights, expected " f"{EXPECTED_KEY_COUNT}"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_cache_only_rebuild_is_bit_identical(mesh_device, state_dict, tmp_path):
    """(b) of the gate: ``state_dict={}`` + a populated cache reproduces every device tensor exactly.

    Model A is built from the checkpoint with a cache path (which writes the cache), hashed, and
    then dropped before model B is built from ``{}`` and the same cache, so only one 8 B-parameter
    model is resident at a time.

    Uses a **2-layer** stack: cache-only mode is a property of ``ttnn.as_tensor``'s
    ``cache_file_name`` plumbing, which is per-tensor and identical in every layer, so 2 layers
    exercise all nine per-layer weight kinds plus all three global ones (21 tensors) at a twentieth
    of the load time. The 32-layer completeness of the *key* audit is
    :func:`test_no_missing_and_no_unused_keys`'s job, and P8 re-runs this one at ``(4,8)`` where the
    sharded cache is what is actually at risk (``R-017`` / Appendix F.10).
    """
    objs = TestFactory.setup_test(mesh_device)
    cache = tmp_path / "cache"
    cache.mkdir()
    kwargs = dict(
        mesh_config=objs["mesh_config"],
        ccl_manager=objs["ccl_manager"],
        max_seq_len=128,
        num_layers=2,
        tensor_cache_path=str(cache),
        with_lm_head=True,
    )

    model_a = Model(mesh_device, objs["hf_config"], state_dict, **kwargs)
    hashes_a = _hash_all(model_a)
    del model_a

    model_b = Model(mesh_device, objs["hf_config"], {}, **kwargs)
    hashes_b = _hash_all(model_b)

    assert set(hashes_a) == set(hashes_b), (
        f"cache-only rebuild holds a different tensor set: " f"{sorted(set(hashes_a) ^ set(hashes_b))}"
    )
    differing = sorted(k for k in hashes_a if hashes_a[k] != hashes_b[k])
    logger.info(
        f"[G-WEIGHTS] cache-only rebuild: {len(hashes_a)} device tensors compared by SHA-256, "
        f"{len(differing)} differ"
    )
    assert not differing, (
        f"[G-WEIGHTS] cache-only rebuild is NOT bit-identical for {len(differing)} tensors: " f"{differing}"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_device_weights_match_the_checkpoint(mesh_device, state_dict):
    """Every sampled device weight equals its checkpoint tensor through the module's own ladder.

    This is the assertion the counts cannot make. A model whose layer 17 was built from layer 16's
    sub-dict holds exactly 291 tensors of exactly the right shapes and passes every audit above; the
    only way to see it is to compare **values**, per key, against the checkpoint.

    The expected tensor is built by replaying each loader's transformation:

    * norm gains: ``reshape(1, 1, -1, 32)``, bf16 (``tt/rms_norm.py``);
    * ``mlp.*`` and ``self_attn.{v,o}_proj``: ``transpose(-1,-2)``, ``weight_dtype``;
    * ``self_attn.{q,k}_proj``: ``reverse_permute`` **first**, then transpose — the Meta RoPE
      swizzle ``load_attention_weights`` applies (``DEC-033``). Comparing without it is the same
      0.9475-PCC error the ``G-ATTN`` negative control measures, so this doubles as a check that the
      swizzle is applied exactly once (``DEC-039``: ``ModelArgs.load_state_dict`` must NOT also
      permute);
    * ``lm_head``: ``transpose(0,1)``;
    * ``model.embed_tokens``: unsqueezed only, ROW_MAJOR bf16, no quantisation of layout.
    """
    from models.tt_transformers.tt.load_checkpoints import reverse_permute

    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    weight_dtype = ttnn.bfloat8_b
    head_dim = hf_config.head_dim

    model = Model(
        mesh_device,
        hf_config,
        state_dict,
        mesh_config=objs["mesh_config"],
        ccl_manager=objs["ccl_manager"],
        max_seq_len=128,
        weight_dtype=weight_dtype,
        with_lm_head=True,
    )
    device_tensors = model.named_device_tensors()

    def _expected(key, w):
        if key.endswith("layernorm.weight") or key == "model.norm.weight":
            return quantize_like_device(w.float().reshape(1, 1, -1, ttnn.TILE_SIZE), ttnn.bfloat16)
        if key == "model.embed_tokens.weight":
            return quantize_like_device(w.float().reshape(1, 1, -1, hf_config.hidden_size), ttnn.bfloat16).reshape(
                1, 1, -1, hf_config.hidden_size
            )
        if key == "lm_head.weight":
            return quantize_like_device(w.float().transpose(0, 1).unsqueeze(0).unsqueeze(0), weight_dtype)
        t = w.float()
        if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.k_proj.weight"):
            t = reverse_permute(t, t.shape[0] // head_dim, t.shape[0], t.shape[1])
        return quantize_like_device(t.transpose(-1, -2).unsqueeze(0).unsqueeze(0), weight_dtype)

    sampled = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in SAMPLED_LAYERS:
        sampled += [k for k in device_tensors if k.startswith(f"model.layers.{i}.")]

    for key in sampled:
        got = ttnn.to_torch(ttnn.get_device_tensors(device_tensors[key])[0]).float()
        want = _expected(key, state_dict[key])
        assert got.shape == want.shape, f"{key}: device shape {tuple(got.shape)} != {tuple(want.shape)}"
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, msg=lambda m, k=key: f"{k}: {m}")
    logger.info(
        f"[G-WEIGHTS] {len(sampled)} device weights (layers {SAMPLED_LAYERS} + the 3 global ones) "
        f"are bit-exactly the checkpoint's, through each loader's own transpose / Meta-swizzle / "
        f"dtype ladder"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_meta_renaming_is_caught_by_the_audit(mesh_device, state_dict, expect_error):
    """Negative control: the HF->Meta key rename the recipe suggested must be caught, not absorbed.

    ``BRINGUP_RECIPE.md:762-764`` and ``03_OUTLINE.md`` §3.3 both prescribe running the checkpoint
    through ``models/tt_transformers/tt/load_checkpoints.py:800`` ``map_hf_to_meta_keys``. Every
    module in this package strips **HF** names (``substate(sd, "mlp")``), so after that rename every
    ``substate`` returns ``{}`` — and with a populated ``tensor_cache_path`` that is not an error at
    all: the modules load whatever the cache holds. This is the exact failure mode ``G-WEIGHTS``
    exists to catch, and it is why ``ModelArgs.load_state_dict`` does not do the rename
    (``DEC-039``).

    Asserts all three lines of defence fire: the audit reports every key missing *and* every key
    unused, the tripwire flags the naming, and construction refuses rather than proceeding.
    """
    from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys

    objs = TestFactory.setup_test(mesh_device)
    args = ModelArgs(mesh_device, weights_path=hf_model_path(), hf_config=objs["hf_config"])

    meta_sd = map_hf_to_meta_keys(dict(state_dict))
    assert state_dict_uses_meta_keys(meta_sd), "the tripwire missed a Meta-renamed state dict"

    missing, unused = args.audit_state_dict_keys(meta_sd)
    logger.info(
        f"[G-WEIGHTS] negative control (map_hf_to_meta_keys applied): {len(missing)} missing, "
        f"{len(unused)} unused of {len(meta_sd)} — e.g. missing "
        f"{sorted(missing)[:2]}, unused {sorted(unused)[:2]}"
    )
    assert len(missing) == EXPECTED_KEY_COUNT, (
        f"the audit found only {len(missing)} missing keys after a full Meta rename; it would let a "
        f"partially-renamed checkpoint through"
    )
    assert len(unused) == EXPECTED_KEY_COUNT, f"the audit found only {len(unused)} unused keys"

    # And no cache path -> the modules must refuse rather than build on nothing (DEC-038).
    with expect_error(AssertionError, "cache"):
        Model(
            mesh_device,
            objs["hf_config"],
            meta_sd,
            mesh_config=objs["mesh_config"],
            ccl_manager=objs["ccl_manager"],
            max_seq_len=128,
            num_layers=1,
        )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_weight_cache_path_carries_the_mesh_shape(mesh_device, tmp_path, monkeypatch):
    """``R-017``: the mesh shape must be a path segment, and the dtype must not share a directory.

    ``ttnn.as_tensor`` caches the **already-sharded** per-device tensor. A ``(1,1)`` cache replayed
    on ``(4,8)`` therefore hands every chip the full unsharded weight; nothing downstream notices,
    and it presents as "one layer runs on garbage" two phases later at ``G-MESH-KV``
    (Appendix F.10). ``models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75`` puts the shape
    in the path for exactly this reason.

    Host-only apart from reading ``mesh_device.shape``; no checkpoint needed, so it runs on a
    weightless machine too.
    """
    monkeypatch.setenv("LLAMA31_8B_TTNN_CACHE", str(tmp_path))
    hf_config = llama_hf_config(llama_config_dims())
    args = ModelArgs(mesh_device, weights_path=str(tmp_path), hf_config=hf_config)

    rows, cols = tuple(mesh_device.shape)
    p8 = args.weight_cache_path(ttnn.bfloat8_b)
    p16 = args.weight_cache_path(ttnn.bfloat16)
    logger.info(f"[G-WEIGHTS] weight_cache_path @ {(rows, cols)}: bf8_b -> {p8} | bf16 -> {p16}")

    assert f"{rows}x{cols}" in p8.parts, f"mesh shape {rows}x{cols} is not a segment of {p8} (R-017)"
    assert str(rows * cols) + "dev" in "".join(p8.parts), f"device count missing from {p8}"
    assert p8 != p16, "bf8_b and bf16 caches share a directory; a dtype switch would read the wrong tree"
    assert p8.is_dir() and p16.is_dir(), "weight_cache_path must create the directory it returns"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_state_dict_prefix_and_audit_refusals(mesh_device, tmp_path, expect_error):
    """``get_state_dict_prefix`` names real keys, and its refusals are asserts, not best effort."""
    hf_config = llama_hf_config(llama_config_dims())
    args = ModelArgs(mesh_device, weights_path=str(tmp_path), hf_config=hf_config)

    assert ModelArgs.get_state_dict_prefix("self_attn", 3) == "model.layers.3.self_attn."
    assert ModelArgs.get_state_dict_prefix("mlp", 0) == "model.layers.0.mlp."
    assert ModelArgs.get_state_dict_prefix("layer", 31) == "model.layers.31."
    assert ModelArgs.get_state_dict_prefix("input_layernorm", 7) == "model.layers.7.input_layernorm."
    assert ModelArgs.get_state_dict_prefix("embedding") == "model.embed_tokens."
    assert ModelArgs.get_state_dict_prefix("norm") == "model.norm."
    assert ModelArgs.get_state_dict_prefix("lm_head") == "lm_head."

    # Every prefix must actually select something out of the expected key set — a prefix that
    # matches nothing is the silent bug this helper exists to prevent.
    keys = {k: None for k in args.expected_state_dict_keys()}
    for name, idx in (("self_attn", 3), ("mlp", 0), ("layer", 31), ("input_layernorm", 7)):
        pref = ModelArgs.get_state_dict_prefix(name, idx)
        assert substate(keys, pref.rstrip(".")), f"prefix {pref!r} selects no key"

    with expect_error(AssertionError, "unknown module"):
        ModelArgs.get_state_dict_prefix("feed_forward", 0)
    with expect_error(AssertionError, "not a per-layer module"):
        ModelArgs.get_state_dict_prefix("lm_head", 0)
    with expect_error(AssertionError, "per-layer"):
        ModelArgs.get_state_dict_prefix("mlp")
    with expect_error(AssertionError, "checkpoint directory"):
        ModelArgs(mesh_device, weights_path="", hf_config=hf_config)
