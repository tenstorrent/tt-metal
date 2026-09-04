# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Real-checkpoint weight loading. Gate: `G-WEIGHTS`.

This gate catches the failure mode where a renamed or unconsumed key means **a layer quietly runs
on random weights** — or, worse, on a correctly-shaped tensor that has been transformed once too
often. Every assertion here is **bit-exact** (`rtol = atol = 0`), never PCC: recipe §2.5 measured a
completely wrong head→column map still scoring PCC 0.99890, and a transpose or a Meta swizzle
applied twice is exactly that class of bug — it produces a plausible tensor, so correlation is the
wrong instrument (`BRINGUP_RECIPE.md:1408-1409`).

What it proves, in the recipe's own three parts (`BRINGUP_RECIPE.md:1404-1411`):

* **(a) no missing and no silently-unused keys.** The checkpoint's key set — read from
  `model.safetensors.index.json`, so all 32 layers are covered without loading 16 GB — must equal
  `ModelArgs.expected_state_dict_keys()` exactly. Both differences are printed.
* **(b) a cache-only rebuild is bit-identical.** Build once with the checkpoint and a
  `tensor_cache_path`; build again with an **empty** `state_dict` and the same path; every device
  tensor must be SHA-256-identical. The runner depends on this branch
  (`BRINGUP_RECIPE.md:929-937`).
* **(c) every device weight is bit-exact against the checkpoint** *through* the loader's transpose,
  the Q/K Meta swizzle and the dtype ladder. Not a sample: all twelve tensors of a one-layer model,
  because a per-tensor check is also the honest proof that every key was **consumed** — an ignored
  key cannot produce a matching device tensor.

**Negative controls, three of them:**

1. **Meta-renamed keys** (`map_hf_to_meta_keys`, `models/tt_transformers/tt/load_checkpoints.py:800`)
   — every expected key must go missing and the model must refuse to build. This is the recipe's
   "bypass `map_hf_to_meta_keys` and every key must go missing" control, inverted to match this
   package's actual key convention (`DEC-046`): the mapping is what this package does **not** do, so
   applying it is what must break.
2. **Double Meta swizzle** — Q/K pre-`reverse_permute`d in the state dict, so the loader's own
   swizzle lands twice. The device tensor must stop being bit-equal. This is what proves part (c) is
   sensitive to a transform applied twice, which is the whole reason it is bit-exact and not PCC
   (`DEC-047`).
3. **Cache written at another dtype** — a bf16 cache must not satisfy a bf8_b build. The cache path
   carries the dtype and the mesh shape for exactly this reason (`DEC-048`).

**Input distribution / reference dtype policy.** Not applicable in the usual sense and stated rather
than omitted: the inputs *are* the real checkpoint tensors at their stored dtype (`bfloat16`,
`bringup_log/00_MODEL_CARD.md` §2), and the "reference" is the same tensor pushed through
`quantize_like_device` — the package's one quantiser (`DEC-007`) — with **no** fp32 detour, so the
comparison cannot be spoiled by double rounding.

Run:
    HF_MODEL=... pytest models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py -x -q
"""

import hashlib
import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links
from models.demos.llama31_8b_d_p.tests.test_factory import (
    GALAXY_MESH_SHAPE,
    bundled_config_path,
    galaxy_device_params,
    hf_model_path,
    llama_config_dims,
    prefill_topology,
    quantize_like_device,
    requires_galaxy,
    requires_hf_reference,
    requires_ring_fabric,
)
from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
from models.demos.llama31_8b_d_p.tt.config import MeshConfig
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs, torch_dtype_of
from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys, reverse_permute

WEIGHT_DTYPE = ttnn.bfloat8_b  # `DEC-022`
# One layer is enough for the per-tensor bit-exactness and the cache-only rebuild: the loader code
# is identical for every layer, and part (a) covers all 32 layers by key set. Building 32 layers
# with real weights twice would cost ~15 minutes of host I/O to re-prove the same code path.
N_LAYERS = 1

# Replicated across the whole mesh rather than TP-sharded (`tt/embedding.py`, `DEC-024`), and large
# enough that hashing all 32 identical copies costs 33.6 GB of D2H per pass. See
# `_all_device_hashes` in the P8 arm.
_REPLICATED_VOCAB_TABLES = ("model.embed_tokens.weight",)

# The keys a one-layer model consumes, in the order they are checked, with the transform each one
# goes through. `swizzle` marks the two the Meta RoPE convention rewrites at load
# (`tt/attention/weights.py`); `transpose` marks HF `[out, in]` -> ttnn `[in, out]`.
_LAYER0 = "model.layers.0."
_WEIGHT_PLAN = (
    # (checkpoint key, transpose?, swizzle?, on-device dtype)
    ("model.embed_tokens.weight", False, False, ttnn.bfloat16),
    (_LAYER0 + "input_layernorm.weight", False, False, ttnn.bfloat16),
    (_LAYER0 + "post_attention_layernorm.weight", False, False, ttnn.bfloat16),
    (_LAYER0 + "self_attn.q_proj.weight", True, True, WEIGHT_DTYPE),
    (_LAYER0 + "self_attn.k_proj.weight", True, True, WEIGHT_DTYPE),
    (_LAYER0 + "self_attn.v_proj.weight", True, False, WEIGHT_DTYPE),
    (_LAYER0 + "self_attn.o_proj.weight", True, False, WEIGHT_DTYPE),
    (_LAYER0 + "mlp.gate_proj.weight", True, False, WEIGHT_DTYPE),
    (_LAYER0 + "mlp.up_proj.weight", True, False, WEIGHT_DTYPE),
    (_LAYER0 + "mlp.down_proj.weight", True, False, WEIGHT_DTYPE),
    ("model.norm.weight", False, False, ttnn.bfloat16),
    ("lm_head.weight", True, False, WEIGHT_DTYPE),
)


def _checkpoint_key_set():
    """Every key in the checkpoint, from the index — no tensor data read."""
    with open(os.path.join(hf_model_path(), "model.safetensors.index.json")) as f:
        return set(json.load(f)["weight_map"])


def _load_subset():
    """Only the tensors a one-layer model needs: embed + layer 0 + final norm + lm_head."""
    return ModelArgs._load_safetensors(
        hf_model_path(),
        prefixes=("model.embed_tokens.", _LAYER0, "model.norm.", "lm_head."),
    )


def _model_tensors(model):
    """`{name: ttnn.Tensor}` for every weight a one-layer `Model` holds, keyed by checkpoint key."""
    layer = model.layers[0]
    return {
        "model.embed_tokens.weight": model.embedding.weight,
        _LAYER0 + "input_layernorm.weight": layer.input_layernorm.tt_weight,
        _LAYER0 + "post_attention_layernorm.weight": layer.post_attention_layernorm.tt_weight,
        _LAYER0 + "self_attn.q_proj.weight": layer.self_attn.weights.q_proj,
        _LAYER0 + "self_attn.k_proj.weight": layer.self_attn.weights.k_proj,
        _LAYER0 + "self_attn.v_proj.weight": layer.self_attn.weights.v_proj,
        _LAYER0 + "self_attn.o_proj.weight": layer.self_attn.weights.o_proj,
        _LAYER0 + "mlp.gate_proj.weight": layer.mlp.gate_proj,
        _LAYER0 + "mlp.up_proj.weight": layer.mlp.up_proj,
        _LAYER0 + "mlp.down_proj.weight": layer.mlp.down_proj,
        "model.norm.weight": model.norm.tt_weight,
        "lm_head.weight": model.lm_head.weight,
    }


def _expected_device_tensor(key, tensor, *, transpose, swizzle, dtype, head_dim):
    """The checkpoint tensor put through **exactly** the loader's transforms, then quantised.

    Deliberately re-implemented here rather than imported: a helper shared with the loader would
    make this test tautological. The steps are the ones the loader's own docstrings claim, so a
    mismatch means the claim is wrong — which is the point.
    """
    t = tensor
    if swizzle:
        n_heads = t.shape[0] // head_dim
        t = reverse_permute(t, n_heads, t.shape[0], t.shape[1])
    if transpose:
        t = t.transpose(-1, -2)
    if key.endswith("layernorm.weight") or key == "model.norm.weight":
        # `tt/rms_norm.py` reshapes the gain to (1, 1, hidden/32, 32) and stores it ROW_MAJOR.
        t = t.reshape((1, 1, -1, ttnn.TILE_SIZE))
        return quantize_like_device(t, dtype)
    if key == "model.embed_tokens.weight":
        # The table stays 2D ROW_MAJOR on device (`tt/embedding.py`); quantise it as a 4D view.
        return quantize_like_device(t[None, None], dtype)[0, 0]
    return quantize_like_device(t.unsqueeze(0).unsqueeze(0), dtype)


def _read_device(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


def _sha256(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def _build_model(mesh_device, hf, state_dict, *, cache_path=None, weight_dtype=WEIGHT_DTYPE, ccl_manager=None):
    """Build a one-layer `Model`. `ccl_manager` is required at TP > 1 (the P8 arm)."""
    return Model(
        mesh_device,
        hf,
        state_dict,
        ccl_manager=ccl_manager,
        mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        weight_dtype=weight_dtype,
        tensor_cache_path=str(cache_path) if cache_path else None,
        max_seq_len=ttnn.TILE_SIZE,
        n_layers=N_LAYERS,
        with_lm_head=True,
    )


# ---------------------------------------------------------------------------------------------
# (a) no missing, no unused — all 32 layers, from the index alone
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
def test_no_missing_and_no_unused_keys():
    """The checkpoint's key set must equal exactly what this package consumes."""
    hf = llama_config_dims()

    class _FakeMesh:
        shape = (1, 1)

    args = ModelArgs(_FakeMesh(), hf_config=hf)
    expected = args.expected_state_dict_keys()
    actual = _checkpoint_key_set()

    missing = sorted(expected - actual)
    unused = sorted(actual - expected)
    logger.info(
        f"[G-WEIGHTS] checkpoint keys={len(actual)} expected={len(expected)} "
        f"missing={len(missing)} unused={len(unused)}"
    )
    logger.info(f"[G-WEIGHTS] missing keys: {missing}")
    logger.info(f"[G-WEIGHTS] unused (silently ignored) keys: {unused}")
    assert not missing, f"the model expects keys the checkpoint does not have: {missing}"
    assert not unused, (
        f"the checkpoint carries keys nothing consumes: {unused} — either a module is silently "
        f"dropping a weight or the expected-key list is stale"
    )


@requires_hf_reference
def test_meta_key_mapping_negative_control():
    """**Control 1:** `map_hf_to_meta_keys` must make every expected key disappear (`DEC-046`)."""
    hf = llama_config_dims()

    class _FakeMesh:
        shape = (1, 1)

    expected = ModelArgs(_FakeMesh(), hf_config=hf).expected_state_dict_keys()
    mapped = set(map_hf_to_meta_keys({k: None for k in _checkpoint_key_set()}))

    overlap = expected & mapped
    logger.info(
        f"[G-WEIGHTS] control: after map_hf_to_meta_keys, {len(mapped)} keys, of which "
        f"{len(overlap)} are still names this package consumes; examples of the renaming: "
        f"{sorted(mapped)[:3]}"
    )
    assert not overlap, (
        f"map_hf_to_meta_keys left {len(overlap)} keys this package would still consume, so the "
        f"control does not discriminate: {sorted(overlap)[:5]}"
    )


@requires_hf_reference
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_model_refuses_meta_renamed_state_dict(mesh_device, expect_error):
    """**Control 1, on device:** a Meta-renamed state dict must fail loud, not build on `None`s."""
    hf = llama_config_dims()
    renamed = map_hf_to_meta_keys({k: torch.zeros(1) for k in _checkpoint_key_set()})
    with expect_error(ValueError, "tensor_cache_path"):
        _build_model(mesh_device, hf, renamed)


# ---------------------------------------------------------------------------------------------
# (c) every device weight bit-exact against the checkpoint, through the loader's transforms
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_device_weights_are_bit_exact_vs_checkpoint(mesh_device):
    """All twelve weights of a one-layer model, `rtol = atol = 0`."""
    hf = llama_config_dims()
    state_dict = _load_subset()
    logger.info(
        f"[G-WEIGHTS] loaded {len(state_dict)} checkpoint tensors; stored dtype "
        f"{torch_dtype_of(state_dict)} (no fp32 detour: the same tensor object drives both sides)"
    )

    model = _build_model(mesh_device, hf, state_dict)
    device = _model_tensors(model)
    head_dim = hf["hidden_size"] // hf["num_attention_heads"]

    for key, transpose, swizzle, dtype in _WEIGHT_PLAN:
        expected = _expected_device_tensor(
            key, state_dict[key], transpose=transpose, swizzle=swizzle, dtype=dtype, head_dim=head_dim
        )
        actual = _read_device(device[key])
        assert tuple(actual.shape) == tuple(
            expected.shape
        ), f"{key}: device {actual.shape} vs expected {expected.shape}"
        max_delta = (actual - expected).abs().max().item()
        logger.info(
            f"[G-WEIGHTS] {key:<48} shape={tuple(actual.shape)} dtype={dtype.name:<10} "
            f"transpose={int(transpose)} swizzle={int(swizzle)} max|delta|={max_delta:.3e} "
            f"sha256={_sha256(actual)[:16]}"
        )
        assert torch.equal(actual, expected), (
            f"{key} is not bit-exact through the loader (max|delta| {max_delta:.3e}). A transpose "
            f"or the Meta swizzle applied a different number of times than documented would look "
            f"exactly like this, and PCC would not see it."
        )


@requires_hf_reference
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_double_meta_swizzle_is_caught(mesh_device):
    """**Control 2:** pre-swizzled Q/K must stop matching, and V/O must be unaffected."""
    hf = llama_config_dims()
    head_dim = hf["hidden_size"] // hf["num_attention_heads"]
    state_dict = dict(_load_subset())
    for name in ("q_proj", "k_proj"):
        key = f"{_LAYER0}self_attn.{name}.weight"
        t = state_dict[key]
        state_dict[key] = reverse_permute(t, t.shape[0] // head_dim, t.shape[0], t.shape[1])

    model = _build_model(mesh_device, hf, state_dict)
    device = _model_tensors(model)
    clean = _load_subset()

    for key, transpose, swizzle, dtype in _WEIGHT_PLAN:
        if not key.startswith(_LAYER0 + "self_attn."):
            continue
        expected = _expected_device_tensor(
            key, clean[key], transpose=transpose, swizzle=swizzle, dtype=dtype, head_dim=head_dim
        )
        equal = torch.equal(_read_device(device[key]), expected)
        logger.info(f"[G-WEIGHTS] control: double-swizzle {key:<48} bit-equal to the clean load = {equal}")
        if swizzle:
            assert not equal, f"{key} survived a double reverse_permute bit-identical — the check is blind"
        else:
            assert equal, f"{key} changed although it is not swizzled — the control perturbed the wrong tensors"


# ---------------------------------------------------------------------------------------------
# (b) cache-only rebuild
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_cache_only_rebuild_is_bit_identical(mesh_device, tmp_path):
    """Build with the checkpoint + a cache path, then with an **empty** state dict + the same path."""
    hf = llama_config_dims()
    args = ModelArgs(mesh_device, hf_config=hf)
    cache_path = args.weight_cache_path(WEIGHT_DTYPE, cache_root=tmp_path)
    logger.info(f"[G-WEIGHTS] cache path: {cache_path.name} (dtype and mesh shape are both in it, DEC-048)")

    first = _build_model(mesh_device, hf, _load_subset(), cache_path=cache_path)
    first_hashes = {key: _sha256(_read_device(t)) for key, t in _model_tensors(first).items()}
    del first

    cached = _build_model(mesh_device, hf, {}, cache_path=cache_path)
    second_hashes = {key: _sha256(_read_device(t)) for key, t in _model_tensors(cached).items()}

    for key in first_hashes:
        logger.info(f"[G-WEIGHTS] cache-only {key:<48} {first_hashes[key][:16]} vs {second_hashes[key][:16]}")
        assert first_hashes[key] == second_hashes[key], f"{key} differs after a cache-only rebuild"
    files = sorted(p.name for p in cache_path.rglob("*.tensorbin"))
    logger.info(f"[G-WEIGHTS] cache-only rebuild: {len(first_hashes)} tensors identical; {len(files)} cache files")


@requires_hf_reference
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_cache_written_at_another_dtype_is_not_reused(mesh_device, tmp_path):
    """**Control 3:** a bf16 cache must not satisfy a bf8_b build — the dtype is in the path."""
    hf = llama_config_dims()
    args = ModelArgs(mesh_device, hf_config=hf)
    bf16_path = args.weight_cache_path(ttnn.bfloat16, cache_root=tmp_path)
    bf8_path = args.weight_cache_path(ttnn.bfloat8_b, cache_root=tmp_path)
    assert bf16_path != bf8_path, "the two dtypes resolved to the same cache directory"

    _build_model(mesh_device, hf, _load_subset(), cache_path=bf16_path, weight_dtype=ttnn.bfloat16)
    bf16_files = sorted(p.name for p in bf16_path.rglob("*.tensorbin"))
    bf8_files = sorted(p.name for p in bf8_path.rglob("*.tensorbin"))
    logger.info(
        f"[G-WEIGHTS] control: a bf16 build wrote {len(bf16_files)} cache files into "
        f"{bf16_path.name} and {len(bf8_files)} into {bf8_path.name}"
    )
    assert bf16_files, "the bf16 build wrote no cache at all — the control proves nothing"
    assert not bf8_files, (
        f"a bf16 build populated the bf8_b cache directory ({bf8_files[:3]}), so a bf8_b build "
        f"could read bf16-derived tensors — the dtype is supposed to separate them (DEC-048)"
    )
    # ttnn additionally suffixes each file with its own dtype and layout, so even a shared
    # directory could not cross-load; both defences are recorded rather than assumed.
    assert all("_dtype_BFLOAT16_" in name for name in bf16_files), sorted(set(bf16_files))[:3]


# ---------------------------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_weight_cache_path_refuses_the_checkpoint_dir(mesh_device, expect_error, monkeypatch):
    """`$HF_MODEL` already holds two foreign tilized caches, so the template's fallback is refused."""
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    args = ModelArgs(mesh_device, hf_config=llama_config_dims())
    with expect_error(ValueError, "refuses to fall back to the checkpoint directory"):
        args.weight_cache_path(WEIGHT_DTYPE)


def test_load_state_dict_refuses_meta_conversion(expect_error):
    """`convert_to_meta_format=True` would swizzle Q/K twice (`DEC-047`)."""
    with expect_error(NotImplementedError, "second time"):
        ModelArgs.load_state_dict("/nonexistent", convert_to_meta_format=True)


def test_model_args_refuses_a_transformers_config_object(expect_error):
    """A `transformers` config **object** must be refused, not silently accepted.

    On 5.12.1 it has no `rope_theta` attribute at all and `getattr(cfg, "rope_theta", DEFAULT)`
    returns the DEFAULT (recipe P1 trap 1, measured in `test_reference_model.py`). Refusing the type
    at the constructor is what makes "the config is a dict, everywhere" enforceable rather than a
    convention.
    """
    from transformers import AutoConfig

    class _FakeMesh:
        shape = (1, 1)

    cfg = AutoConfig.from_pretrained(os.path.dirname(bundled_config_path()))
    assert not hasattr(cfg, "rope_theta"), "this transformers version has rope_theta; re-check the trap"
    with expect_error(TypeError, "raw config.json dict"):
        ModelArgs(_FakeMesh(), hf_config=cfg)


@requires_hf_reference
def test_state_dict_prefixes_match_the_checkpoint():
    """`get_state_dict_prefix` must name prefixes the checkpoint actually has.

    Nothing in P6 calls it — `Model` splits with literal `substate` prefixes — but it is part of the
    contract `bringup_log/03_OUTLINE.md` §2.3 pins and the interface P7's runtime and P10's adapter
    inherit from the templates (`models/demos/gpt_oss_d_p/tt/model_config.py:175`). An untested
    accessor is a claim, so it is checked here against the real key set rather than left to be
    discovered wrong two phases later.
    """

    class _FakeMesh:
        shape = (1, 1)

    args = ModelArgs(_FakeMesh(), hf_config=llama_config_dims())
    cases = {
        args.get_state_dict_prefix("self_attn", 0): "model.layers.0.self_attn.",
        args.get_state_dict_prefix("mlp", 31): "model.layers.31.mlp.",
        args.get_state_dict_prefix("", 7): "model.layers.7.",
        args.get_state_dict_prefix("norm"): "model.norm.",
        args.get_state_dict_prefix("embed_tokens"): "model.embed_tokens.",
    }
    for produced, expected in cases.items():
        assert produced == expected, f"get_state_dict_prefix produced {produced!r}, expected {expected!r}"

    keys = _checkpoint_key_set()
    for prefix in cases:
        matches = [k for k in keys if k.startswith(prefix)]
        logger.info(f"[G-WEIGHTS] prefix {prefix!r} matches {len(matches)} checkpoint keys")
        assert matches, f"get_state_dict_prefix produced {prefix!r}, which matches no checkpoint key"


# =============================================================================================
# (d) `G-WEIGHTS`, the P8 extension: the cache-only rebuild **at TP=8**, where the cache is
# actually sharded.
#
# `BRINGUP_RECIPE.md:1808-1810`: "`ttnn.as_tensor` caches the already-sharded tensor, so a stale or
# wrong-shape cache presents as 'one layer runs on garbage' and is first visible here, not at
# `G-WEIGHTS`". The `(1,1)` arm above cannot see it for two reasons, both structural:
#
#   1. at TP=1 there is nothing to shard, so the persisted tensor is the full-width one and any
#      sharding bug is absent from the file rather than baked into it;
#   2. `_read_device` reads **device 0 only**, which at `(1,1)` is the whole tensor and at `(4,8)`
#      is 1/8 of it — so the arm below hashes **every one of the 32 device tensors** and a cache
#      that reconstituted the shards in the wrong order would still pass a device-0 check.
#
# The mesh shape is in the cache path (`DEC-048`), so a `(1,1)` cache cannot be picked up here by
# accident; `test_cache_written_at_another_dtype_is_not_reused` above is the same argument for the
# dtype. This arm adds the mesh-shape half.
# =============================================================================================
@requires_hf_reference
@requires_galaxy
@requires_ring_fabric
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_cache_only_rebuild_is_bit_identical_at_tp8(mesh_device, tmp_path):
    """Every one of the 32 device shards must be SHA-256-identical after a cache-only rebuild."""
    hf = llama_config_dims()
    args = ModelArgs(mesh_device, hf_config=hf)
    cache_path = args.weight_cache_path(WEIGHT_DTYPE, cache_root=tmp_path)
    assert cache_path.name.endswith(f"_{GALAXY_MESH_SHAPE[0]}x{GALAXY_MESH_SHAPE[1]}"), (
        f"the mesh shape must be in the cache path so a (1,1) cache cannot be reused at TP=8; got " f"{cache_path.name}"
    )
    logger.info(f"[G-WEIGHTS] TP=8 cache path: {cache_path.name}")

    def _all_device_hashes(model):
        """`{key: {device_index: sha256}}` over every device shard, except as noted below.

        `model.embed_tokens.weight` is **replicated, not TP-sharded** (`tt/embedding.py`,
        `DEC-024`), so each of the 32 devices holds the whole `[128256, 4096]` table — 1.05 GB
        each, 33.6 GB of device-to-host transfer per pass, twice, to re-prove a tensor the mesh
        does not shard. For that one tensor the first and last device are hashed (which is what
        makes the *replication* falsifiable) and the per-device claim is the `(1,1)` arm's
        (`DEC-087`). Every other tensor, `lm_head.weight` included, is hashed on all 32.
        """
        out = {}
        for key, tensor in _model_tensors(model).items():
            shards = ttnn.get_device_tensors(tensor)
            picked = (0, len(shards) - 1) if key in _REPLICATED_VOCAB_TABLES else range(len(shards))
            out[key] = {dev: _sha256(ttnn.to_torch(shards[dev]).float()) for dev in picked}
        return out

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=prefill_topology())
    first = _build_model(mesh_device, hf, _load_subset(), cache_path=cache_path, ccl_manager=ccl)
    first_hashes = _all_device_hashes(first)
    del first

    cached = _build_model(mesh_device, hf, {}, cache_path=cache_path, ccl_manager=ccl)
    second_hashes = _all_device_hashes(cached)

    n_shards = 0
    sharded, replicated = [], []
    for key in first_hashes:
        assert set(first_hashes[key]) == set(second_hashes[key]), f"{key}: the device set changed"
        n_shards += len(first_hashes[key])
        distinct = len(set(first_hashes[key].values()))
        (sharded if distinct > 1 else replicated).append(key)
        for dev, a in first_hashes[key].items():
            b = second_hashes[key][dev]
            assert a == b, f"{key} shard {dev} differs after a cache-only rebuild: {a[:16]} vs {b[:16]}"
        logger.info(
            f"[G-WEIGHTS] TP=8 cache-only {key:<48} {len(first_hashes[key])} shards hashed, "
            f"{distinct} distinct -> {'SHARDED' if distinct > 1 else 'replicated'}, "
            f"dev0 {first_hashes[key][0][:16]}"
        )

    files = sorted(p.name for p in cache_path.rglob("*.tensorbin"))
    logger.info(
        f"[G-WEIGHTS] TP=8 cache-only rebuild: {n_shards} device shards over "
        f"{len(first_hashes)} tensors, all SHA-256-identical. Genuinely sharded ({len(sharded)}): "
        f"{sharded}. Replicated ({len(replicated)}): {replicated}. {len(files)} cache files."
    )
    # The whole point of the P8 extension: if nothing were actually sharded, this arm would be the
    # (1,1) arm again with more devices.
    assert sharded, (
        "no weight came back with more than one distinct shard hash, so nothing is sharded at "
        "TP=8 and this arm is not testing what it claims to. Check MeshConfig.column_parallel."
    )
    # The replication claim, made falsifiable rather than assumed: the vocab table's first and last
    # device must agree, and every projection must NOT.
    for key in _REPLICATED_VOCAB_TABLES:
        hashes = set(first_hashes[key].values())
        assert len(hashes) == 1, f"{key} is documented as replicated (DEC-024) but its shards differ"
    for key in ("model.layers.0.self_attn.q_proj.weight", "lm_head.weight"):
        assert key in sharded, f"{key} is column-parallel and must differ across the TP columns"
