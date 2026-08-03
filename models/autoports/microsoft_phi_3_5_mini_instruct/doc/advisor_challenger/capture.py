"""Phi-3.5 hooks for advisor-challenger's fixed capture template."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[5]
TEMPLATE = ROOT / ".agents/skills/advisor-challenger/scripts/capture_template.py"
spec = importlib.util.spec_from_file_location("advisor_challenger_capture_template", TEMPLATE)
fixed = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fixed)


def _config():
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import _config
    return _config()


def _synthetic_state_dict(config):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import _synthetic_state
    return _synthetic_state(config)


def _build(device):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
        _page_table, _positions, _to_tt_decode,
    )
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy

    if fixed.SHIPPED_POLICY != {"policy_name": "final"}:
        raise ValueError(f"unsupported frozen policy: {fixed.SHIPPED_POLICY}")
    config = _config()
    decoder = fixed.OptimizedDecoder.from_state_dict(
        _synthetic_state_dict(config), hf_config=config, layer_idx=fixed.LAYER_IDX,
        mesh_device=device, batch=fixed.BATCH, max_context=128,
        policy=replace(OptimizationPolicy(), advisor_rope_l1_chain=False),
    )

    # The capture template forbids dynamic tensor.memory_config() queries: layout is
    # deliberately unknown while the advisor assigns it. The shipped producer above
    # explicitly requests L1 height sharding, so mirror _decode_rope with that declared
    # phase config instead of querying the traced value.
    def capture_decode_rope(self, query, key, current_positions, *, use_long_rope):
        import ttnn
        cos_table = self.long_cos_decode if use_long_rope else self.short_cos_decode
        sin_table = self.long_sin_decode if use_long_rope else self.short_sin_decode
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.reshape(
            ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT),
            [1, 1, self.batch, self.head_dim],
        )
        sin = ttnn.reshape(
            ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT),
            [1, 1, self.batch, self.head_dim],
        )
        cos = ttnn.transpose(cos, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.transpose(sin, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        return (
            ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
            ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
        )

    decoder._decode_rope = MethodType(capture_decode_rope, decoder)
    hidden = torch.randn(
        fixed.BATCH, 1, config.hidden_size,
        generator=torch.Generator().manual_seed(52), dtype=torch.bfloat16,
    )
    tt_hidden = _to_tt_decode(hidden, device)
    page_table = _page_table(fixed.BATCH, 128, device, permute=True)
    current_positions = _positions([127] * fixed.BATCH, device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    kwargs = dict(
        key_cache=key_cache, value_cache=value_cache, page_table=page_table,
        current_positions=current_positions, use_long_rope=False,
    )
    return decoder, kwargs, tt_hidden


_DECODER = None
_KWARGS = None


def decode(hidden):
    # Mirror the shipped decode path. The pinned tracer has no handler for
    # paged_fused_update_cache, so that terminal state mutation is the sole
    # omitted op; its measured share is recorded as unreachable.
    import ttnn
    self = _DECODER
    key_cache = _KWARGS["key_cache"]
    value_cache = _KWARGS["value_cache"]
    page_table = _KWARGS["page_table"]
    current_positions = _KWARGS["current_positions"]
    residual = ttnn.to_memory_config(hidden, self.residual_memory_config)
    normalized = self._norm_decode(residual, self.weights["input_norm"])
    fused = self._linear_decode(normalized, "qkv", self.qkv_memory_config, self.attention_compute_kernel_config)
    query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
        fused, num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    query, key = self._decode_rope(query, key, current_positions, use_long_rope=False)
    value = ttnn.to_memory_config(value, self._fused_cache_value_memory_config())
    attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        query, key_cache, value_cache, cur_pos_tensor=current_positions,
        page_table_tensor=page_table, scale=self.scale,
        program_config=None, compute_kernel_config=self.attention_compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
    attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
    attended = ttnn.to_memory_config(attended, self.residual_memory_config)
    projected = self._linear_decode(attended, "o_proj", self.residual_memory_config, self.attention_compute_kernel_config)
    projected = ttnn.reshape(projected, [1, 1, self.batch, self.hidden_size])
    return self._mlp_decode(ttnn.add(residual, projected, memory_config=self.residual_memory_config))


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    fixed._record_traced_dtypes(str(Path(__file__).parent / "shard_advise" / fixed.LAYER_KIND))
    return (hidden,)


if __name__ == "__main__":
    import argparse
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize-report")
    args = ap.parse_args()
    if args.finalize_report:
        with open(args.finalize_report) as fh:
            report = json.load(fh)
        report.update({
            "traced_weight_dtypes": fixed.SHIPPED_DTYPES,
            "capture_policy_source": fixed._incumbent.get("shipped_policy_source"),
            "capture_batch": fixed.BATCH,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        with open(args.finalize_report, "w") as fh:
            json.dump(report, fh, indent=2)
        raise SystemExit(0)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        make_inputs(device)
        print(f"capture target builds: kind={fixed.LAYER_KIND} idx={fixed.LAYER_IDX} batch={fixed.BATCH}")
    finally:
        ttnn.close_mesh_device(device)
