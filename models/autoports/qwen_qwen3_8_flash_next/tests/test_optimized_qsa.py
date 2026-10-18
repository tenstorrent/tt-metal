# SPDX-License-Identifier: Apache-2.0
"""Attention-only diagnostics; all device commands are scheduled by the stage owner.

Example:
  TT_QSA_LENGTHS=5 TT_QSA_WEIGHTS=real python_env/bin/python -m pytest -s \
    models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_qsa.py -k hf_parity

TT_QSA_LENGTHS can select 3,4,5,31,32,33,65,2053,2055,262141,262144.
Long full-HF attention is deliberately not presented as tractable; the full
context sparse-reference harness remains required before stage completion.
"""

import json
import os
from pathlib import Path

import pytest
import torch

CHECKPOINT = Path(
    os.environ.get(
        "TT_QSA_CHECKPOINT",
        "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540",
    )
)
LAYER = 3
PREFIX = f"model.language_model.layers.{LAYER}.self_attn."


def load_attention_reference(real=True):
    from safetensors import safe_open
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention, Qwen4ExpTextRotaryEmbedding

    cfg = Qwen4ExpTextConfig(**json.loads((CHECKPOINT / "config.json").read_text())["text_config"])
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        reference = Qwen4ExpTextAttention(cfg, LAYER)
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    generator = torch.Generator().manual_seed(379)
    stats = {}
    # Read only the attention tensors, not the 335 GiB whole checkpoint.
    for name in reference.state_dict():
        key = PREFIX + name
        with safe_open(CHECKPOINT / index[key], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(key)
        mean, std = value.float().mean().item(), value.float().std().item()
        stats[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "mean": mean, "std": std}
        state[name] = value if real else torch.randn(value.shape, generator=generator).mul_(std).add_(mean).bfloat16()
    reference.load_state_dict(state, assign=True)
    reference.eval()
    return cfg, reference, Qwen4ExpTextRotaryEmbedding(cfg), state, stats


def pcc(expected, actual):
    a, b = expected.flatten().double(), actual.flatten().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    a, b = a - a.mean(), b - b.mean()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def reference_forward(reference, rotary, x):
    positions = torch.arange(x.shape[1]).reshape(1, -1).expand(x.shape[0], -1)
    cos_sin = rotary(x, positions)
    causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()[None, None]
    # QSAIndexer indexes the batch dimension explicitly; broadcasting is not sufficient.
    causal = causal.expand(x.shape[0], -1, -1, -1)
    mask = torch.where(causal, 0.0, torch.finfo(x.dtype).min).to(x.dtype)
    with torch.no_grad():
        return reference(x, position_embeddings=cos_sin, attention_mask=mask)[0]


def sparse_reference_forward(reference, rotary, x, *, chunk_size=64, progress=False):
    """HF modules with bounded sparse attention; never creates [S,S] tensors.

    Pool/index scores follow installed HF line-for-line. Main attention gathers
    selected keys and evaluates the same BF16 matmul/FP32 softmax/BF16 matmul as
    eager_attention_forward. Future keys may be preprojected, but are excluded
    before top-k. A control test compares this harness against unmodified HF.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch, total, _ = x.shape
    positions = torch.arange(total).reshape(1, -1).expand(batch, -1)
    cos, sin = rotary(x, positions)
    keys = torch.empty(batch, 2, total, 256, dtype=x.dtype)
    values = torch.empty_like(keys)
    raw = torch.empty(batch, total, 128, dtype=x.dtype)
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            part = x[:, start:end]
            key = reference.k_norm(reference.k_proj(part).reshape(batch, -1, 2, 256)).transpose(1, 2)
            key = apply_rotary_pos_emb(key, cos=cos[:, start:end], sin=sin[:, start:end])
            keys[:, :, start:end] = key
            values[:, :, start:end] = reference.v_proj(part).reshape(batch, -1, 2, 256).transpose(1, 2)
            raw[:, start:end] = reference.indexer.index_qk_proj(part)[..., 512:]
        blocks = total // 4
        if blocks:
            pool = raw[:, : blocks * 4].reshape(batch, blocks, 4, 128).float().mean(2).to(x.dtype)
            pool = reference.indexer.k_layernorm(pool)
            pool = apply_rotary_pos_emb(
                pool.unsqueeze(1), cos=cos[:, : blocks * 4 : 4], sin=sin[:, : blocks * 4 : 4]
            ).squeeze(1)
        output = torch.empty_like(x)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            count = end - start
            part = x[:, start:end]
            qg = reference.q_proj(part).reshape(batch, count, 24, 512)
            query = reference.q_norm(qg[..., :256]).transpose(1, 2)
            query = apply_rotary_pos_emb(query, cos=cos[:, start:end], sin=sin[:, start:end]).transpose(1, 2)
            iq = reference.indexer.index_qk_proj(part)[..., :512].reshape(batch, count, 4, 128)
            iq = reference.indexer.q_layernorm(iq)
            iq = apply_rotary_pos_emb(iq, cos=cos[:, start:end], sin=sin[:, start:end], unsqueeze_dim=2)
            visible = torch.arange(start + 1, end + 1).reshape(1, count, 1)
            complete = visible // 4
            if blocks:
                scores = torch.matmul(iq.permute(0, 2, 1, 3).float(), pool[:, None].float().transpose(-1, -2))
                scores = scores.relu().sum(dim=1) / (128**0.5)
                scores.masked_fill_(torch.arange(blocks).reshape(1, 1, -1) >= complete, float("-inf"))
                selected = scores.topk(min(512, blocks), dim=-1).indices
                tokens = (selected[..., None] * 4 + torch.arange(4)).flatten(-2)
                valid = (selected < complete).repeat_interleave(4, dim=-1)
            else:
                tokens = torch.empty(batch, count, 0, dtype=torch.int64)
                valid = torch.empty(batch, count, 0, dtype=torch.bool)
            tail = (complete * 4 + torch.arange(3)).expand(batch, -1, -1)
            tokens = torch.cat((tokens, tail), dim=-1)
            valid = torch.cat((valid, tail < visible), dim=-1)
            tokens = tokens.clamp(max=total - 1)
            bi = torch.arange(batch)[:, None, None, None]
            hi = torch.arange(2)[None, None, :, None]
            selected_k = keys[bi, hi, tokens[:, :, None, :]]
            selected_v = values[bi, hi, tokens[:, :, None, :]]
            q = query.reshape(batch, count, 2, 12, 256)
            scores = torch.matmul(q, selected_k.transpose(-1, -2)) * reference.scaling
            scores.masked_fill_(~valid[:, :, None, None, :], torch.finfo(x.dtype).min)
            probability = scores.softmax(-1, dtype=torch.float32).to(x.dtype)
            attended = torch.matmul(probability, selected_v).reshape(batch, count, 6144)
            output[:, start:end] = reference.o_proj(attended * qg[..., 256:].reshape(batch, count, 6144).sigmoid())
            if progress and (start == 0 or end == total or start // 4096 != (start + chunk_size) // 4096):
                print(f"QSA_HF_SPARSE_PROGRESS {end}/{total}", flush=True)
    return output


def test_qsa_sparse_reference_control():
    """Validate the long-context harness against the unmodified HF attention."""
    cfg, reference, rotary, _, _ = load_attention_reference(real=True)
    for length in (3, 5, 33, 2053):
        x = (
            torch.randn(1, length, cfg.hidden_size, generator=torch.Generator().manual_seed(length))
            .mul_(0.1)
            .bfloat16()
        )
        expected = reference_forward(reference, rotary, x)
        actual = sparse_reference_forward(reference, rotary, x, chunk_size=32)
        correlation = pcc(expected, actual)
        assert correlation >= 0.99999, (length, correlation)
        print(f"QSA_SPARSE_REFERENCE_CONTROL length={length} pcc={correlation:.9f}", flush=True)


def make_tt(reference_state, cfg, mesh_device, batch, length):
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedQSA as QSA

    capacity = max(2048, (length + 255) // 256 * 256)
    qsa = QSA.from_state_dict(
        reference_state,
        hf_config=cfg,
        layer_idx=LAYER,
        mesh_device=mesh_device,
        batch_size=batch,
        max_context=capacity,
        max_requests=batch if os.environ.get("TT_QSA_REQUEST_SLOTS") == "1" else None,
    )
    pages = capacity // 32
    ownership = torch.randperm(batch * pages, generator=torch.Generator().manual_seed(32)).reshape(batch, pages)
    page_table = ttnn.from_torch(
        ownership.to(torch.int32), device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    positions = ttnn.from_torch(
        torch.zeros(batch, dtype=torch.int32), device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    return qsa, page_table, positions


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("length", [int(n) for n in os.environ.get("TT_QSA_LENGTHS", "5,33,65,2053").split(",")])
@pytest.mark.parametrize("batch", [int(n) for n in os.environ.get("TT_QSA_BATCHES", "1,2").split(",")])
def test_qsa_hf_parity(mesh_device, length, batch, tmp_path):
    import ttnn

    real = os.environ.get("TT_QSA_WEIGHTS", "real") == "real"
    cfg, reference, rotary, state, stats = load_attention_reference(real=real)
    total = min(length + 1, cfg.max_position_embeddings)
    decode_index = total - 1
    x = torch.randn(batch, total, 2560, generator=torch.Generator().manual_seed(10)).mul_(0.1).bfloat16()
    reference_cache = os.environ.get("TT_QSA_REFERENCE_CACHE")
    if reference_cache:
        cached = torch.load(reference_cache, map_location="cpu", weights_only=True)
        assert cached["shape"] == list(x.shape)
        assert cached["seed"] == 10 and cached["real_weights"] == real
        assert cached["checkpoint_revision"] == CHECKPOINT.name
        expected = cached["output"]
    else:
        expected = (
            sparse_reference_forward(reference, rotary, x, progress=True)
            if total > 4096
            else reference_forward(reference, rotary, x)
        )
    qsa, page_table, positions = make_tt(state, cfg, mesh_device, batch, total)
    input_tt = ttnn.from_torch(
        x[:, :length].unsqueeze(0), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    request_slots = None
    if os.environ.get("TT_QSA_REQUEST_SLOTS") == "1":
        host_slots = torch.zeros(batch, 32, dtype=torch.int32)
        host_slots[:, 0] = torch.arange(batch)
        request_slots = ttnn.from_torch(host_slots, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    actual = qsa.prefill_forward(input_tt, current_pos=positions, page_table=page_table, slots=request_slots)
    actual_cpu = ttnn.to_torch(actual).squeeze(0)
    prefill_pcc = pcc(expected[:, :length], actual_cpu)
    assert prefill_pcc >= 0.995, prefill_pcc
    next_input = ttnn.from_torch(
        x[:, decode_index : decode_index + 1].unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    next_position = ttnn.from_torch(
        torch.full((batch,), decode_index, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    # Repeated current-token writes are idempotent; raw index keys remain paged.
    qsa.decode_forward(next_input, current_pos=next_position, page_table=page_table, slots=request_slots)
    ttnn.synchronize_device(mesh_device)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decoded = qsa.decode_forward(next_input, current_pos=next_position, page_table=page_table, slots=request_slots)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    repeated = []
    try:
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            repeated.append(ttnn.to_torch(decoded).squeeze(0))
        decode_pcc = pcc(expected[:, decode_index : decode_index + 1], repeated[-1])
        assert decode_pcc >= 0.995, decode_pcc
        assert torch.equal(repeated[0], repeated[1]) and torch.equal(repeated[1], repeated[2])
    finally:
        ttnn.release_trace(mesh_device, trace)
    evidence = {
        "layer": LAYER,
        "length": length,
        "batch": batch,
        "real_weights": real,
        "request_slots": request_slots is not None,
        "prefill_pcc": prefill_pcc,
        "traced_decode_pcc": decode_pcc,
        "determinism_bitwise": True,
        "page_permutation": True,
        "stats": stats,
    }
    evidence_path = (
        Path(__file__).resolve().parents[1]
        / "doc/optimized_decoder"
        / f"qsa_context_{length}_B{batch}{'_slots' if request_slots is not None else ''}.json"
    )
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "stats"}))
    print(f"QSA_EVIDENCE={evidence_path}")


def test_qsa_q_gate_checkpoint_packing():
    """CPU-only contract check protects the non-global Q/gate split."""
    cfg, reference, _, state, _ = load_attention_reference(real=True)
    x = torch.randn(1, 2, cfg.hidden_size, generator=torch.Generator().manual_seed(5)).bfloat16()
    packed = reference.q_proj(x).reshape(1, 2, 24, 512)
    weight = state["q_proj.weight"].reshape(24, 512, 2560)
    query = torch.nn.functional.linear(x, weight[:, :256].reshape(6144, 2560))
    gate = torch.nn.functional.linear(x, weight[:, 256:].reshape(6144, 2560))
    assert torch.equal(query.reshape(1, 2, 24, 256), packed[..., :256])
    assert torch.equal(gate.reshape(1, 2, 24, 256), packed[..., 256:])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("prefix", [33, 2047])
def test_qsa_trace_request_slots(mesh_device, prefix):
    """Changed token, slot, page table and position payloads reuse one trace."""
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedQSA as QSA

    cfg, reference, rotary, state, _ = load_attention_reference(real=True)
    batch = 2
    x = torch.randn(batch, prefix + 1, 2560, generator=torch.Generator().manual_seed(113)).mul_(0.1).bfloat16()
    expected_full = reference_forward(reference, rotary, x)
    expected = expected_full[:, -1:]
    qsa = QSA.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=LAYER,
        mesh_device=mesh_device,
        batch_size=batch,
        max_context=2048,
        max_requests=batch,
    )
    pages_cpu = torch.randperm(128, generator=torch.Generator().manual_seed(18)).reshape(batch, 64).int()
    slots_cpu = torch.zeros(batch, 32, dtype=torch.int64)
    slots_cpu[:, 0] = torch.arange(batch)
    positions_cpu = torch.zeros(batch, 32, dtype=torch.int64)

    def upload(value, dtype, layout):
        return ttnn.from_torch(value.contiguous(), device=mesh_device, dtype=dtype, layout=layout)

    pages = upload(pages_cpu, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    slots = upload(slots_cpu, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    positions = upload(positions_cpu, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    # A fresh public decode must initialize its own routing extent; no prior
    # prefill is required. Then a long prefill must not contaminate decode's
    # compile-time store range at the final valid context position.
    first = upload(x[:, :1].unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    fresh = qsa.decode_forward(first, current_pos=positions, page_table=pages, slots=slots)
    fresh_pcc = pcc(expected_full[:, :1], ttnn.to_torch(fresh).squeeze(0))
    assert fresh_pcc >= 0.995, fresh_pcc
    assert qsa.route_sequence == 1
    print(f"QSA_FRESH_DECODE prefix={prefix} pcc={fresh_pcc:.12f}", flush=True)
    pref = upload(x[:, :prefix].unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    qsa.prefill_forward(pref, current_pos=positions, page_table=pages, slots=slots)
    next_x = upload(x[:, -1:].unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    positions_cpu[:, 0] = prefix

    def refresh(value, destination, dtype, layout):
        host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)
        ttnn.copy_host_to_device_tensor(host, destination)

    refresh(positions_cpu, positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    qsa.decode_forward(next_x, current_pos=positions, page_table=pages, slots=slots)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = qsa.decode_forward(next_x, current_pos=positions, page_table=pages, slots=slots)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for permutation in (torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 1])):
            refresh(x[permutation, -1:].unsqueeze(0), next_x, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            refresh(pages_cpu[permutation], pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            refresh(slots_cpu[permutation], slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            correlation = pcc(expected[permutation], ttnn.to_torch(output).squeeze(0))
            assert correlation >= 0.995, correlation
            print(f"QSA_SLOT_TRACE permutation={permutation.tolist()} trace_id={trace} pcc={correlation:.9f}")
    finally:
        ttnn.release_trace(mesh_device, trace)
