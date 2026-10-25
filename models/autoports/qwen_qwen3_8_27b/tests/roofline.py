# SPDX-License-Identifier: Apache-2.0
"""Reproducible whole-layer algorithmic roofline estimates (not hardware counters)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'doc/functional_decoder'
# One p300c ASIC:120 physical cores,1.35GHz, HiFi4 divides4096 FLOP/cycle by4.
# Ten dispatch/storage-reserved cores leave110 available for this runtime.
PEAK_FLOPS = 120 * 4096 * 1.35e9 / 4
# p300c has two ASICs and1024GB/s aggregate GDDR6:512GB/s for the one selected ASIC.
PEAK_DRAM = 512e9


def calculate(kind, batch, prefill_us, decode_us):
    s, context, h, intermediate = 4096, 4224, 5120, 17408
    mlp = 3 * h * intermediate
    common_scalar = 10*h + 5*intermediate + 2*h  # norms, SiLU/product, residuals
    if kind == 'full_attention':
        matrices = mlp + h*(2*6144 + 2*1024 + 6144)
        # QK+AV causal pairs: two dot products,2FLOPs per multiply/add.
        attention = 2 * 24 * 256 * s * (s+1)
        softmax = 5 * 24 * s * (s+1) / 2  # exp/div/comparison each counted as one scalar op
        scalar = common_scalar + 5*(6144+1024) + 4*64*28 + 4*6144
        useful = batch * (2*s*matrices + attention + softmax + s*scalar)
        # Compulsory weight, current prefix KV read, one-token KV write.
        dram = batch*(2*matrices + (context+1)*4096 + 2*h*2)
        traffic = 'BF16 dense weights once per independent batch plane of the [B,1,H] matmul geometry; prefix KV read once, one KV token write, residual input/output. Excludes intermediate spills, repeated reads within a plane, norms and page metadata: estimated lower-bound traffic, not measured bus utilization.'
    else:
        matrices = mlp + h*(10240 + 6144 + 48 + 48 + 6144)
        recurrence = 7*48*128*128  # decay, two state-vector dots, rank-one multiply/add
        scalar = common_scalar + 8*10240 + 4*4096 + 5*6144 + 8*48
        useful = batch*s*(2*matrices + recurrence + scalar)
        dram = batch*(2*matrices + 2*48*128*128*4 + 2*32*10240*2 + 2*h*2)
        traffic = 'BF16 dense weights once per independent batch plane of the [B,1,H] matmul geometry; FP32 recurrent state and physically padded BF16 convolution state read/write, residual input/output. Excludes intermediate spills, repeated reads within a plane and small norms/gates: estimated lower-bound traffic, not measured bus utilization.'
    return dict(prefill_device_us=prefill_us, decode_device_us=decode_us,
                prefill_useful_flops=useful, decode_dram_bytes=dram,
                peak_flops_per_s=PEAK_FLOPS, peak_dram_bytes_per_s=PEAK_DRAM,
                prefill_flops_pct=100*useful/(PEAK_FLOPS*prefill_us/1e6) if prefill_us else None,
                decode_dram_pct=100*dram/(PEAK_DRAM*decode_us/1e6),
                peak_basis='One Blackhole p300c ASIC:120 physical cores at1.35GHz,4096 FLOP/cycle/core divided by4 for HiFi4;512GB/s DRAM. Runtime compute grid110 cores. No multi-device aggregation.',
                traffic_assumptions=traffic,
                flops_assumptions='Dense active work, no MoE/sparsity;2 FLOPs per multiply/add; causal attention pairs or recurrent updates plus approximate scalar norms/gates/activation work. Scalar transcendentals counted as one operation. No padding counted as useful work.',
                source='single-layer Tracy; whole-layer device firmware timestamp envelope including gaps')


def main():
    result = {}
    for kind, short in [('linear_attention','linear'), ('full_attention','full')]:
        result[kind] = {}
        for batch in [1,8,16,32]:
            name = f'{short}_b{batch}' + ('_complete' if short=='linear' and batch==1 else '')
            if short == 'linear' and batch == 32:
                name += '_decode'
            path = ROOT / 'tracy' / name / 'whole_layer_timing.json'
            if not path.exists():
                continue
            d = json.loads(path.read_text())['phases']
            entry = calculate(kind,batch,d.get('prefill', {}).get('device_us'),d['decode']['device_us'])
            if 'prefill' not in d:
                entry['missing_reasons'] = {'prefill_device_us': 'Batch32 supplemental decode profile avoids Tracy32K source-location limit. Batch1 warmed prefill is measured.'}
            entry['evidence'] = [f'tracy/{name}/whole_layer_timing.json',f'tracy/{name}/ops.csv',f'profile_{name}.json']
            result[kind][str(batch)] = entry
    (ROOT / 'performance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
