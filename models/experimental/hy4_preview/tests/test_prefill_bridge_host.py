# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The host half of the prefill -> decode hand-over (tt/prefill_bridge.py), tenstorrent/tt-metal#65893.

* The slot arithmetic of the prefill tensors, and the host view of them (HostCaches) against
  tt/kv_layout.py: a row written at (chip, row) by the prefill stripe is read back as its position.
* The row check the device test relies on (row_report): on synthetic rows and on HF's own cache
  rows of the golden prompts it passes a noisy copy and fails every displacement a wrong stripe
  or slot produces (a neighbour, a tile, a block, a round, two chips swapped, another layer),
  while the PCC over the whole tensor stays high for several of them.
* The decode golden (reference/hf_harness.py record_decode_golden) is what a decode after a
  prefill must reproduce: the reference decode step (reference/attention_decode.py) run over HF's
  cached rows [0, p) with HF's input for position p gives HF's full-sequence attention output of
  that row and, where the context is longer than the top-k, HF's picks. With the checkpoint's
  weights, layers 0-5, below and above 2048 tokens.

Pure CPU, no device, no ttnn. The golden tests skip when the goldens are not recorded:
    python -m models.experimental.hy4_preview.reference.hf_harness --decode-steps 6 --seq-len 1022 \
        --prompt-file tech_reports/LLMs/llms.md         (and --seq-len 5118)
"""

import pytest
import torch

from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention
from models.experimental.hy4_preview.reference.attention_decode import DecodeCache, Hy4AttentionDecode
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig, full_layers
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.kv_cache import HY4_INDEXER_LAYERS
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout
from models.experimental.hy4_preview.tt.prefill_bridge import (
    HostCaches,
    format_report,
    index_slot,
    kvpe_slot,
    row_pcc,
    row_report,
)

LAYOUT = BlockCyclicLayout()
GOLDEN_PROMPTS = (1022, 5118)
ROPE = 64


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


# ---------------------------------------------------------------------------------------------
# Slots and the host view
# ---------------------------------------------------------------------------------------------
def test_slots_are_the_prefill_stacks():
    types = full_layers(78)
    assert tuple(n for n, kind in enumerate(types) if kind == "full") == HY4_INDEXER_LAYERS
    for user in (0, 3, 7):
        for layer in (0, 1, 5, 77):
            assert kvpe_slot(user, layer, 78) == user * 78 + layer
            assert index_slot(user, layer, types) == user * 21 + HY4_INDEXER_LAYERS.index(layer)
    # a runtime built for 6 layers strides by 6 and by its 3 full layers
    six = types[:6]
    assert kvpe_slot(2, 5, 6) == 17 and index_slot(2, 5, six) == 2 * 3 + 2
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        index_slot(0, 2, types)  # shared layer
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        kvpe_slot(0, 6, 6)


def _stack_stripe(rows, seq_len):
    """Natural rows [N, W] -> [32, seq_len / 32, W] by the prefill stack's own formula, written out:
    chip = (p // 128) % 32, row = (p // 4096) * 128 + p % 128."""
    out = torch.zeros(32, seq_len // 32, rows.shape[-1])
    p = torch.arange(rows.shape[0])
    out[(p // 128) % 32, (p // 4096) * 128 + p % 128] = rows
    return out


def test_host_caches_read_positions_back():
    users, layers, seq_len, n = 2, 6, 16384, 9000
    types = tuple(full_layers(layers))
    full = [layer for layer in range(layers) if types[layer] == "full"]
    g = torch.Generator().manual_seed(0)
    latent = torch.randn(users, layers, n, 16, generator=g)
    keys = torch.randn(users, len(full), n, 8, generator=g)
    kvpe = torch.stack(
        [_stack_stripe(latent[u, layer], seq_len) for u in range(users) for layer in range(layers)], dim=1
    )
    index = torch.stack(
        [_stack_stripe(keys[u, rank], seq_len) for u in range(users) for rank in range(len(full))], dim=1
    )
    assert kvpe.shape == (32, users * layers, seq_len // 32, 16)
    host = HostCaches(
        kvpe=kvpe,
        index=index,
        num_layers=layers,
        indexer_types=types,
        destripe=lambda slabs, length: LAYOUT.merge(slabs, length),
        unpack=lambda rows: rows,
    )
    for u in range(users):
        for layer in range(layers):
            assert torch.equal(host.latent(u, layer, n), latent[u, layer])
            if host.is_full(layer):
                assert torch.equal(host.index_key(u, layer, n), keys[u, full.index(layer)])
    # and the decode cache's own striping of those rows is the prefill slab, row for row
    ours = LAYOUT.split(latent[1, 3], slots=seq_len // 32)
    assert torch.equal(ours, kvpe[:, kvpe_slot(1, 3, layers)])


# ---------------------------------------------------------------------------------------------
# The row check
# ---------------------------------------------------------------------------------------------
def _displacements(rows):
    """name -> the rows as a wrong hand-over would return them (same shape)."""
    n = rows.shape[0]
    out = {}
    for d in (1, 32, 128, 4096):
        if d < n:
            out[f"shifted by {d}"] = torch.roll(rows, d, dims=0)
    stripes = LAYOUT.split(rows)
    swapped = stripes.clone()
    swapped[[3, 4]] = stripes[[4, 3]]
    out["chips 3 and 4 swapped"] = LAYOUT.merge(swapped, n)
    if n > 4096:
        rounds = stripes.clone()
        rounds[:, :128], rounds[:, 128:256] = stripes[:, 128:256], stripes[:, :128]
        out["rounds 0 and 1 swapped"] = LAYOUT.merge(rounds, n)
    return out


def _ok(report, min_row):
    return report["misplaced"] == 0 and report["min_row"] >= min_row


def test_row_pcc():
    a = torch.randn(5, 64)
    assert torch.allclose(row_pcc(a, a), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(row_pcc(a, -2 * a + 3), -torch.ones(5, dtype=torch.float64))
    assert row_pcc(torch.ones(1, 8), torch.randn(1, 8)).item() == 0.0


@pytest.mark.parametrize("n", [1022, 5118])
def test_row_report_on_rows_that_resemble_one_another(n):
    """Rows with a large common part: a displaced cache keeps a whole-tensor PCC near 0.9, which
    is why the hand-over is checked row by row."""
    g = torch.Generator().manual_seed(n)
    common = torch.randn(1, 576, generator=g)
    ref = 3.0 * common + torch.randn(n, 576, generator=g)
    good = ref + 0.03 * torch.randn(n, 576, generator=g)
    report = row_report(ref, good)
    assert _ok(report, 0.99), format_report(report)
    for name, moved in _displacements(good).items():
        report = row_report(ref, moved)
        if "rounds" not in name:  # swapping rounds also moves the empty rows above the prompt in
            assert report["pcc"] > 0.85, f"{name}: the whole-tensor PCC was expected to stay high, {report['pcc']:.4f}"
        assert not _ok(report, 0.99), f"{name} was not caught: {format_report(report)}"
        assert report["misplaced"] > 0, f"{name}: {format_report(report)}"
    other_layer = 3.0 * common + torch.randn(n, 576, generator=g)
    report = row_report(ref, other_layer)
    assert not _ok(report, 0.99) and report["min_row"] < 0.95, format_report(report)


def test_row_report_leaves_twin_rows_out():
    """A reference row repeated one block later (a repeated token at layer 0) is no evidence either
    way: the row is not counted as misplaced, and a real displacement of the others still is."""
    g = torch.Generator().manual_seed(4)
    ref = torch.randn(600, 64, generator=g)
    ref[300] = ref[300 - 128]
    good = ref + 0.02 * torch.randn(ref.shape, generator=g)
    report = row_report(ref, good)
    assert report["misplaced"] == 0 and report["twins"] == 2, format_report(report)
    report = row_report(ref, torch.roll(good, 128, dims=0))
    assert report["misplaced"] >= 600 - 128 - 2, format_report(report)


# ---------------------------------------------------------------------------------------------
# The goldens
# ---------------------------------------------------------------------------------------------
def _golden(prompt_len):
    golden = hf_harness.find_decode_golden(hf_harness.DEFAULT_NUM_LAYERS, prompt_len)
    if golden is None:
        pytest.skip(f"no decode golden for a {prompt_len}-token prompt under {hf_harness.GOLDEN_ROOT}")
    return golden


@pytest.mark.parametrize("prompt_len", GOLDEN_PROMPTS)
def test_golden_is_consistent(prompt_len):
    golden = _golden(prompt_len)
    meta, ids = golden["meta"], golden["input_ids"]
    n, steps = meta["prompt_len"], meta["steps"]
    s = n + steps - 1
    assert ids.numel() == n + steps and meta["positions"] == list(range(n - 1, s))
    # HF's cached generation and its full-sequence pass pick the same tokens
    assert torch.equal(golden["logits"].argmax(-1), ids[n:]), (golden["logits"].argmax(-1), ids[n:])
    assert torch.equal(golden["cached_logits_argmax"], ids[n:])
    for layer, (cached, rec) in enumerate(zip(golden["cache"], golden["layers"])):
        assert cached["latent"].shape == (s, 576)
        assert ("index_key" in cached) == (meta["indexer_types"][layer] == "full")
        # the cached k_pass of the decode positions is the full pass's k_pass of those rows (bf16 noise apart)
        score = pcc(cached["latent"][n - 1 :, :512].float(), rec["k_pass"].float())
        assert score > 0.9995, f"layer {layer}: cached k_pass vs the full pass, pcc={score:.6f}"
        dropped = rec["dsa_dropped_rows"]
        if s <= meta["index_topk"]:
            assert dropped == [0] * steps, f"layer {layer}: picks drop keys below the top-k: {dropped}"
        else:
            assert dropped == [p + 1 - meta["index_topk"] for p in meta["positions"]], dropped


@pytest.mark.parametrize("prompt_len", GOLDEN_PROMPTS)
def test_frames_round_trip_on_golden_rows(prompt_len):
    golden = _golden(prompt_len)
    for cached in golden["cache"]:
        latent = cached["latent"].float()
        device = frames.tail_to_device(latent, ROPE)
        assert torch.equal(device[:, :512], latent[:, :512])
        assert torch.equal(frames.tail_from_device(device, ROPE), latent)
        if "index_key" in cached:
            key = cached["index_key"].float()
            back = frames.index_from_device(frames.index_to_device(key, ROPE), ROPE)
            assert (back - key).abs().max() < 1e-4 * key.abs().max()


@pytest.mark.parametrize("prompt_len", GOLDEN_PROMPTS)
def test_row_report_on_golden_rows(prompt_len):
    """HF's own rows of a real prompt, as a correct hand-over would return them (in the device
    frame, rounded the way a cache rounds) and as a wrong one would."""
    golden = _golden(prompt_len)
    n = golden["meta"]["prompt_len"]
    g = torch.Generator().manual_seed(1)
    lowest = {}
    for layer, cached in enumerate(golden["cache"]):
        kinds = {"latent": frames.tail_to_device(cached["latent"][:n].float(), ROPE)}
        if "index_key" in cached:
            kinds["index_key"] = frames.index_to_device(cached["index_key"][:n], ROPE)
        for kind, ref in kinds.items():
            good = ref * (1 + 0.02 * torch.randn(ref.shape, generator=g))  # coarser than BFP8 or FP8 rounding
            report = row_report(ref, good)
            assert _ok(report, 0.99), f"layer {layer} {kind}: {format_report(report)}"
            for name, moved in _displacements(good).items():
                report = row_report(ref, moved)
                assert report["misplaced"] > 0, f"layer {layer} {kind}, {name} not caught: {format_report(report)}"
                lowest[name] = max(lowest.get(name, -1.0), report["pcc"])
            if layer + 1 < len(golden["cache"]) and kind == "latent":
                wrong = frames.tail_to_device(golden["cache"][layer + 1]["latent"][:n].float(), ROPE)
                report = row_report(ref, wrong)
                assert not _ok(report, 0.99), f"layer {layer}: the next layer's rows pass: {format_report(report)}"
    print({name: round(score, 4) for name, score in lowest.items()})


def _reference(layer, full):
    names = [
        f"model.layers.{layer}.self_attn.{n}" + ("" if n == "learnable_sink_param" else ".weight")
        for n in Hy4Attention.PARAMETER_NAMES
    ]
    attention = Hy4Attention(AttentionConfig()).eval()
    with torch.no_grad():
        for name, value in zip(Hy4Attention.PARAMETER_NAMES, hf_harness.checkpoint_tensors(names)):
            getattr(attention, name).copy_(value.reshape(getattr(attention, name).shape))
    indexer = None
    if full:
        keys = list(Hy4Indexer.CHECKPOINT_NAMES.values())
        tensors = hf_harness.checkpoint_tensors([f"model.layers.{layer}.self_attn.indexer.{k}" for k in keys])
        indexer = Hy4Indexer(IndexerConfig()).eval().load_weights(dict(zip(keys, tensors)))
    return Hy4AttentionDecode(attention, indexer)


@pytest.mark.parametrize("prompt_len", GOLDEN_PROMPTS)
def test_reference_decode_over_the_golden_cache(prompt_len):
    """What the device has to do after a prefill, in torch: for every layer and decode position p,
    the reference step over HF's cached rows [0, p) with HF's attention input of row p. It must
    return HF's attention output of that row (HF ran the whole sequence in one pass, in bf16) and,
    on a full layer above the top-k, HF's picks; and append HF's cache rows."""
    golden = _golden(prompt_len)
    meta = golden["meta"]
    topk = meta["index_topk"]
    if not (hf_harness.model_dir() / "model.safetensors.index.json").is_file():
        pytest.skip("checkpoint not found")
    hf_picks = None
    for layer, (cached, rec) in enumerate(zip(golden["cache"], golden["layers"])):
        full = meta["indexer_types"][layer] == "full"
        ref = _reference(layer, full)
        if full:
            hf_picks = rec["topk_indices"].long()
        scores, same = [], []
        for j, p in enumerate(meta["positions"]):
            cache = DecodeCache(
                cached["latent"][:p].float().clone(), cached["index_key"][:p].float().clone() if full else None
            )
            x = rec["attn_in"][j : j + 1].float()
            visible = hf_picks[j][hf_picks[j] <= p]  # HF pads a short context's picks with keys it then masks
            with torch.no_grad():
                out = ref.step(x, cache, picks=None if full else visible, position=p)
            scores.append(pcc(out.out, rec["attn_out"][j].float()))
            if full:
                same.append(len(set(out.picks.tolist()) & set(visible.tolist())) / min(topk, p + 1))
            # the rows the step appended are HF's cache rows of that position
            assert pcc(cache.latent[p], cached["latent"][p].float()) > 0.9998, f"layer {layer} position {p}: latent"
            if full:
                assert pcc(cache.index_keys[p], cached["index_key"][p].float()) > 0.9998, f"layer {layer} {p}: key"
        print(
            f"layer {layer} P{prompt_len}: attn_out pcc min {min(scores):.6f}"
            + (f", picks shared {min(same):.4f}" if same else "")
        )
        assert min(scores) > 0.999, f"layer {layer}: reference decode vs HF attn_out {scores}"
        if full:
            assert min(same) > 0.99, f"layer {layer}: the reference's picks share {same} with HF's"
