# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic real-weight attention projection attribution, never perf evidence.

Uses evidence.decoder_class/decoder_kwargs (set QWEN_DECODER=optimized,
QWEN_OPT_CANDIDATE=packer53 and its frozen QWEN_OPT_POLICY externally).
Reads actual device BFP8 weights and FP32 partials outside any measured region.
No HF decoder/full model or fallback is constructed. Existing gates are unchanged.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer

VARIANTS = {
    'hifi4_pack': ('HiFi4', True), 'hifi4_nopack': ('HiFi4', False),
    'hifi2_pack': ('HiFi2', True), 'hifi2_nopack': ('HiFi2', False),
}
KEYS = ('packed_gdn', 'linear_attn.out_proj.weight')
SEGMENTS = {'q': (0, 2048), 'k': (2048, 4096), 'v': (4096, 10240),
            'z': (10240, 16384), 'b': (16384, 16432), 'a': (16448, 16496)}


def metric(reference, actual):
    assert reference.shape == actual.shape, (reference.shape, actual.shape)
    a, b = reference.double().flatten(), actual.double().flatten()
    aa = a.square().sum()
    ac, bc = a-a.mean(), b-b.mean()
    denom = ac.norm()*bc.norm()
    return {'pcc': float((ac*bc).sum()/denom) if denom > 0 else None,
            'gain': float((a*b).sum()/aa) if aa > 0 else None,
            'relative_rms': float((a-b).norm()/aa.sqrt().clamp_min(1e-30)),
            'max_abs': float((a-b).abs().max()),
            'equal_fraction': float((a == b).double().mean()),
            'bf16_equal_fraction': float((reference.bfloat16() == actual.bfloat16()).double().mean())}


def compare(reference, actual, key):
    if key != 'packed_gdn':
        return {'all': metric(reference, actual)}
    pieces = {name: metric(reference[..., lo:hi], actual[..., lo:hi]) for name, (lo, hi) in SEGMENTS.items()}
    valid = lambda t: torch.cat([t[..., lo:hi] for lo, hi in SEGMENTS.values()], dim=-1)
    return {'all_valid_channels': metric(valid(reference), valid(actual)), **pieces}


def original_weight(weights, key):
    if key != 'packed_gdn':
        return weights[key].T.contiguous().bfloat16()
    packed = torch.cat([weights['linear_attn.in_proj_qkv.weight'], weights['linear_attn.in_proj_z.weight'],
                        F.pad(weights['linear_attn.in_proj_b.weight'], (0, 0, 0, 16)),
                        F.pad(weights['linear_attn.in_proj_a.weight'], (0, 0, 0, 16))], dim=0)
    assert packed.shape == (16512, 5120), packed.shape
    return packed.T.contiguous().bfloat16()


def cpu_projection(x, w, block=512):
    """FP64 reference accumulated from actual stored values, bounded temporary RAM."""
    xf = x.double().reshape(-1, x.shape[-1])
    return torch.cat([xf @ w[:, start:start+block].double() for start in range(0, w.shape[-1], block)], dim=-1)


def fingerprint(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def probe_variant(name, policy, config, weights, capture, sources, user_index):
    # Every variant constructs its policy before setup. No config mutation after creation.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        kwargs = decoder_kwargs()
        kwargs['policy'] = policy
        layer = decoder_class().from_state_dict(weights, hf_config=config, layer_idx=0, mesh_device=mesh, **kwargs)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
        def read(t):
            interleaved = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) if t.is_sharded() else t
            return ttnn.to_torch(interleaved, mesh_composer=composer).detach().cpu().clone()
        for key in KEYS:
            correction = layer.weight_corrections[key]
            assert correction == key+'_residual'
            assert layer.weights[key].dtype == layer.weights[correction].dtype == ttnn.bfloat8_b
            assert key in layer.projection_info and correction in layer.projection_info
            assert layer.projection_info[key]['input_mem'] == layer.projection_info[correction]['input_mem']
            original = original_weight(weights, key)
            logical_n = original.shape[-1]
            assert layer.projection_info[key]['logical_n'] == logical_n
            w0, w1 = (read(layer.weights[k]).float()[:, :logical_n].contiguous() for k in (key, correction))
            assert w0.shape == w1.shape == original.shape
            # These are actual device-dequantized tensors, not quantizer emulation.
            weight_info = {'w0_sha256': fingerprint(w0), 'w1_sha256': fingerprint(w1),
                           'original_bf16_sha256': fingerprint(original),
                           'base_vs_original': metric(original, w0),
                           'pair_vs_original': metric(original, w0+w1)}
            for source in sources:
                field = 'input_norm' if key == 'packed_gdn' else 'gated'
                host_x = capture[source][field][user_index:user_index+1].reshape(1, 1, -1).bfloat16()
                assert host_x.shape[-1] == original.shape[0]
                x = ttnn.from_torch(host_x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                partials = {}
                native_linear = ttnn.linear
                def record_linear(*args, **kw):
                    result = native_linear(*args, **kw)
                    for k in (key, correction):
                        if len(args) > 1 and args[1] is layer.weights[k]:
                            assert k not in partials, ('duplicate partial', k)
                            assert result.dtype == ttnn.float32
                            partials[k] = result
                    return result
                with patch.object(ttnn, 'linear', record_linear):
                    actual = layer._linear(x, key)
                assert set(partials) == {key, correction}, ('Unexpected _linear topology', list(partials))
                assert actual.dtype == ttnn.bfloat16
                # Additional diagnostic FP32 add; production _linear above still supplies final BF16.
                sum_device = ttnn.add(partials[key], partials[correction], dtype=ttnn.float32)
                p0, p1 = (read(partials[k]).reshape(1, -1)[:, :logical_n] for k in (key, correction))
                tt_sum = read(sum_device).reshape(1, -1)[:, :logical_n]
                tt_final = read(actual).reshape(1, logical_n)
                cpu0, cpu1 = cpu_projection(host_x, w0), cpu_projection(host_x, w1)
                cpu_sum = cpu0+cpu1
                original_sum = cpu_projection(host_x, original)
                hf_bf16 = F.linear(host_x.reshape(1, -1), original.T.contiguous())
                sums_of_tt_partials = p0.double()+p1.double()
                comparisons = {
                    'base_partial_vs_dequantized_fp64': (cpu0, p0),
                    'correction_partial_vs_dequantized_fp64': (cpu1, p1),
                    'tt_fp32_sum_vs_dequantized_fp64': (cpu_sum, tt_sum),
                    'tt_fp32_add_vs_cpu_sum_of_tt_partials': (sums_of_tt_partials, tt_sum),
                    'tt_final_vs_cpu_bf16_sum_of_tt_partials': (sums_of_tt_partials.bfloat16(), tt_final),
                    'tt_final_vs_tt_fp32_sum_then_bf16': (tt_sum.bfloat16(), tt_final),
                    'tt_final_vs_dequantized_fp64_then_bf16': (cpu_sum.bfloat16(), tt_final),
                    'dequantized_pair_vs_original_fp64': (original_sum, cpu_sum),
                    'dequantized_pair_bf16_vs_hf_bf16': (hf_bf16, cpu_sum.bfloat16()),
                    'original_fp64_bf16_vs_hf_bf16': (hf_bf16, original_sum.bfloat16()),
                    'tt_final_vs_hf_bf16': (hf_bf16, tt_final),
                }
                row = {'variant': name, 'key': key, 'input_source': source, 'captured_field': field,
                       'capture_step': capture['step'], 'capture_user': capture['users'][user_index],
                       'input_shape': list(host_x.shape), 'input_sha256': fingerprint(host_x),
                       'weight_evidence': weight_info,
                       'comparisons': {label: compare(a, b, key) for label, (a, b) in comparisons.items()}}
                rows.append(row)
                print(name, key, source, row['comparisons']['tt_final_vs_hf_bf16'].get('all_valid_channels', row['comparisons']['tt_final_vs_hf_bf16'].get('all')), flush=True)
                del x, actual, sum_device, partials, p0, p1, tt_sum, tt_final
            del w0, w1, original
        del layer
        gc.collect()
    finally:
        ttnn.close_mesh_device(mesh)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, default=Path('models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/capture41/layer0_step209.pt'))
    ap.add_argument('--user-index', type=int, default=0, help='Index within selected users saved in the capture')
    ap.add_argument('--input-source', choices=('tt', 'hf', 'both'), default='both')
    ap.add_argument('--variant', action='append', choices=tuple(VARIANTS), help='Repeatable; defaults to all four')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    prov = provenance()
    assert prov['decoder_implementation'] == 'optimized', 'Set QWEN_DECODER=optimized'
    assert prov['optimization_candidate'] == 'packer53', 'This probe audits the frozen packer53 implementation'
    base = decoder_kwargs()['policy']
    assert base.get('dram_sharded') and base.get('correction_l1'), 'Use frozen packer53 policy with DRAM sharding/L1 correction'
    assert all(k in base.get('weight_correction', []) for k in KEYS)
    for key in KEYS:
        for k in (key, key+'_residual'):
            geometry = base.get('geometry', {}).get(k, {})
            assert all(geometry.get(n) == value for n, value in [('cores',32),('readers',4),('block',1)]), (k, geometry)
    capture = torch.load(args.capture, weights_only=False, map_location='cpu')
    assert 0 <= args.user_index < len(capture['users'])
    config = Qwen3_5TextConfig(**json.loads((Path(__file__).parents[1]/'doc/hf_config.json').read_text())['text_config'])
    weights = load_layer(0)
    sources = ('tt','hf') if args.input_source == 'both' else (args.input_source,)
    report = {'provenance': prov, 'diagnostic_only_no_perf_claim': True, 'capture': str(args.capture),
              'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              'cpu_reference': 'FP64 accumulation of actual device-dequantized BFP8 operands; original checkpoint BF16 F.linear separately reported',
              'variants': [], 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.variant or tuple(VARIANTS):
        fidelity, pack = VARIANTS[name]
        policy = copy.deepcopy(base) | {'attention_fidelity': fidelity, 'packer_l1_acc': pack}
        report['variants'].append({'name': name, 'policy': policy})
        report['rows'].extend(probe_variant(name, policy, config, weights, capture, sources, args.user_index))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    # No new acceptance threshold: report attribution without waiving decoder gates.


if __name__ == '__main__':
    main()
