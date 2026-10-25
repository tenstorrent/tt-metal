# SPDX-License-Identifier: Apache-2.0
"""Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
"""
from math import ceil

CHIPS = 4
H, I, VOCAB = 5120, 17408, 248320
LINEAR_LAYERS, FULL_LAYERS = 48, 16
PEAK_FLOPS = CHIPS * 120 * 1.35e9 * 4096
PEAK_DRAM = CHIPS * 512e9
PEAK_FLOPS_PER_SECOND = PEAK_FLOPS
PEAK_DRAM_BYTES_PER_SECOND = PEAK_DRAM
PEAK_SOURCE = 'https://docs.tenstorrent.com/aibs/blackhole/p300.html; tests/report_optimized_multichip.py; matmul_device_operation.cpp performance model'
WORK_METHOD = __doc__


def _validate(rows, batch):
    if batch not in (1,8,16,32):
        raise ValueError('Only Stage10 supported capacities1,8,16,32 are modeled')
    if not rows or len(rows)>batch:
        raise ValueError('Require one or more live rows, at most physical capacity')


def prefill_breakdown(rows, batch):
    """Per-invocation useful operations separated for peak-fidelity accounting."""
    _validate(rows,batch)
    dense = attention = scalar = recurrence = terminal = 0
    common = 12*H + 5*I
    linear_matrices = 3*H*I + H*(10240+6144+48+48+6144)
    full_matrices = 3*H*I + H*(2*6144+2*1024+6144)
    linear_scalar = common + 8*10240 + 4*4096 + 5*6144 + 8*48
    full_scalar = common + 5*(6144+1024) + 4*64*28 + 4*6144
    for row in rows:
        start,length = int(row['start']),int(row['length'])
        if start<0 or length<=0 or start+length>262144:
            raise ValueError('Invalid prefill span')
        dense += 2*length*(LINEAR_LAYERS*linear_matrices+FULL_LAYERS*full_matrices)
        pairs = length*start + length*(length+1)//2
        attention += FULL_LAYERS*4*24*256*pairs
        scalar += length*(LINEAR_LAYERS*linear_scalar+FULL_LAYERS*full_scalar)+FULL_LAYERS*5*24*pairs
        recurrence += LINEAR_LAYERS*length*7*48*128*128
        if row.get('final',False):
            terminal += 2*H*VOCAB
            scalar += 5*H
    return {'dense_lofi':dense,'attention_lofi':attention,
            'scalar_hifi4_approx':scalar,'recurrence_hifi4':recurrence,'head_hifi2':terminal}


def prefill_work(rows, batch):
    return sum(prefill_breakdown(rows,batch).values())


def prefill_peak(rows, batch):
    """Ideal work-weighted nominal peak; scalar op ceiling is only a convention.

    Assign scalar/recurrence work HiFi4 peak conservatively for explicit fidelity
    accounting; this is not an SFPU throughput calibration. Report this assumption.
    """
    parts=prefill_breakdown(rows,batch)
    ideal_seconds = ((parts['dense_lofi']+parts['attention_lofi'])/PEAK_FLOPS
                     +2*parts['head_hifi2']/PEAK_FLOPS
                     +4*(parts['scalar_hifi4_approx']+parts['recurrence_hifi4'])/PEAK_FLOPS)
    return sum(parts.values())/ideal_seconds


def _tile(rows, cols, tile_bytes):
    return ceil(rows/32)*ceil(cols/32)*tile_bytes


def _layer_weights(kind):
    mix=4608 if kind=='linear' else 3584
    return CHIPS*sum(_tile(k,n,576) for k,n in
                     [(H,mix),(1536,H),(H,8704),(4352,H)])


def decode_breakdown(rows, batch):
    _validate(rows,batch)
    positions=[int(row['position']) for row in rows]
    if any(p<0 or p>=262144 for p in positions):
        raise ValueError('Invalid decode position')
    weights=LINEAR_LAYERS*_layer_weights('linear')+FULL_LAYERS*_layer_weights('full')
    head=CHIPS*_tile(H,VOCAB//CHIPS,1088)
    # multichip_decoder.fp32_recurrence supports in-place unmasked state.
    # The physical110-core grid splits capacity16/32 into groups of at most9.
    masked = len(rows)<batch
    state_transfers = (7 + (2 if batch>9 else 0)) if masked else (2 + (4 if batch>9 else 0))
    recurrent=LINEAR_LAYERS*batch*state_transfers*48*128*128*4
    conv=LINEAR_LAYERS*batch*2*3*10240*2
    kv=FULL_LAYERS*sum((p+2)*2176 for p in positions)
    snapshots=LINEAR_LAYERS*sum((p+1)%4096==0 for p in positions)*2*(48*128*128*4+3*10240*2)
    # Each native fused out/down writes its DRAM MM result and RS reads it.
    fused_ccl=(64*2*2*CHIPS*_tile(32,H,2048)) if batch>=8 else 0
    # Full hidden is replicated at terminal: write, norm read/write, head read.
    terminal_hidden=4*CHIPS*_tile(32,H,2048)
    logits=2*CHIPS*_tile(32,VOCAB//CHIPS,2048)
    final_norm=CHIPS*_tile(1,H,2048)
    embedding=batch*H*2
    return {'decoder_weights':weights,'head_weights':head,
            'recurrent_state_and_copy':recurrent,'conv_history':conv,'attention_kv':kv,
            'prefix_snapshots':snapshots,
            'native_rs_dram_intermediate':fused_ccl,'terminal_hidden':terminal_hidden,
            'terminal_logits_write_and_sample_read':logits,'final_norm_weight':final_norm,
            'embedding_lookup':embedding}


def decode_work(rows, batch):
    return sum(decode_breakdown(rows,batch).values())
