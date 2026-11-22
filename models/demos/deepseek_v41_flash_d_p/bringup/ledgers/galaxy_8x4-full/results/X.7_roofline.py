# per chip, one 5120-token chunk, layers 0-19 (+L20 kv head), 8x4 torus
T = 5120
chips = 32
rows = T // chips
L = 20
H = 5120
I = 2304
E = 384
k = 6
core_lofi = 5.4e12
cores = 110
hifi2 = cores * core_lofi / 2
lofi = cores * core_lofi
eff_mm = 0.70  # large-matmul fraction (GEMM_FLOPS: 580 of ~700 TF on 130 cores = 0.83; small M lower)
dram = 512e9 * 0.80  # kimi_k27 reference / tt-moe-nappkin: 80% of 512
link = 25e9
links_per_dir = 2


def ms(x):
    return round(x * 1e3, 2)


# experts
pairs = T * k
fl = pairs * 2 * 3 * H * I / chips * L
wbytes = E / chips * 3 * H * I * 0.53 * L
print(
    "experts flop/chip",
    fl / 1e12,
    "TF; compute hifi2",
    ms(fl / (hifi2 * eff_mm)),
    "lofi",
    ms(fl / (lofi * eff_mm)),
    "weights GB",
    wbytes / 1e9,
    "dram ms",
    ms(wbytes / dram),
)
# dispatch: per chip 160 tok * 6 picks * 10240 B (bf16), torus 8x4: X ring avg 2 hops, Y ring 1 hop; 2 dirs x 2 links per axis
b = rows * k * H * 2 * (31 / 32)
x = b * 2 / (2 * links_per_dir * link)
y = b * 1 / (2 * links_per_dir * link)
print(
    "dispatch bytes/chip/layer MB",
    b / 1e6,
    "X-axis ms/layer",
    ms(x),
    "Y",
    ms(y),
    "per chunk peak",
    ms(x * L),
    "at 66%",
    ms(x * L / 0.66),
    "fp8 at 66%",
    ms(x * L / 0.66 * 0.53),
)
# shared expert + router
sf = T * 2 * 3 * H * I / chips * L
sw = 3 * H * I * 1.06 / 4 * L
print("shared flop ms", ms(sf / (hifi2 * eff_mm)), "weights ms", ms(sw / dram))
# attention projections (full weights each chip, 160 rows)
for nm, K, N, calls in [
    ("wq_b", 1280, 32768, L),
    ("wo_b", 8192, 5120, L),
    ("wo_a", 4096, 8 * 1024, L),
    ("q_a", 5120, 1792, L),
    ("kv", 5120, 512, L),
]:
    f = 2 * rows * K * N * calls
    w = K * N * 1.0625 * calls
    print(nm, "compute", ms(f / (hifi2 * eff_mm)), "weights", ms(w / dram), "GB", round(w / 1e9, 2))
# sparse sdpa: 64 heads, 160 q, keys 128 window + 512 topk (18 layers), 128 only (2 layers), d 512
f = sum(2 * 2 * 64 * rows * (128 + (512 if l >= 2 else 0)) * 512 for l in range(L))
kb = sum(rows * (512 if l >= 2 else 0) * 512 * 0.625 for l in range(L))
print("sdpa compute hifi2", ms(f / (hifi2 * 0.5)), "gathered K bytes GB", kb / 1e9, ms(kb / dram))
# rope q: 160x32768 bf16 rd+wr
print("rope q", ms(rows * 32768 * 2 * 2 * L / dram))
# mhc_post: streams 160 x 4*5120 fp32 rd+wr + x
mb = (rows * 4 * H * 4 * 2 + rows * H * 2) * 2 * L
print("mhc_post DRAM", ms(mb / dram), "streams per chip MB", rows * 4 * H * 4 / 1e6)
# indexer score: 32 heads x 160 x 76800 keys (142k/2 + chunk) x 128, 3 calls
f = 2 * 32 * rows * 76800 * 128 * 3
print("indexer score lofi", ms(f / (lofi * eff_mm)), "hifi2", ms(f / (hifi2 * eff_mm)))
# history gather 4-bit: 3 sources, 76800 rows x 320 B arrive per chip, ring all-gather
g = 76800 * 320 * 3
print("gather link ms (8 links in, 70%)", ms(g / (8 * link * 0.7)))
