import torch, ttnn
from dataclasses import dataclass
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import auto_shard_config, shard_config

device = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout
TD = torch.bfloat16


@dataclass(kw_only=True)
class Default:
    legacy_reduction: bool = False
    legacy_rsqrt: bool = False
    use_welford: bool = False


@dataclass(kw_only=True)
class Sharded:
    compute_with_storage_grid_size: object
    subblock_w: int
    block_h: int
    block_w: int
    inplace: bool
    legacy_reduction: bool = False
    legacy_rsqrt: bool = False
    use_welford: bool = False


def sharded_for(t, **over):
    spec = t.memory_config().shard_spec
    bb = spec.grid.bounding_box()
    cfg = Sharded(
        compute_with_storage_grid_size=ttnn.CoreCoord(bb.end.x - bb.start.x + 1, bb.end.y - bb.start.y + 1),
        subblock_w=1,
        block_h=spec.shape[0] // 32,
        block_w=spec.shape[1] // 32,
        inplace=False,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def run(lab, shape, ml, pc_kind, shard=None, expect_refuse=False, mc_override="input", **over):
    torch.manual_seed(0)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(TD)
    if ml == _ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tw = torch.randn(W, dtype=torch.float32).to(TD)
    kw = dict(
        weight=ttnn.from_torch(tw.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    )
    pc = Default(**over) if pc_kind == "default" else (sharded_for(tx, **over) if pc_kind == "sharded" else pc_kind)
    kw["program_config"] = pc
    if mc_override == "input" and ml != _ML.INTERLEAVED:
        kw["memory_config"] = tx.memory_config()
    elif mc_override == "dram":
        kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    try:
        out_t = rms_norm_ttnn(tx, epsilon=1e-12, **kw)
    except Exception as ex:
        print(f"{lab:60s} {'OK-refused' if expect_refuse else 'EXC'} {type(ex).__name__}: {str(ex)[:150]}", flush=True)
        return
    if expect_refuse:
        print(f"{lab:60s} BAD: expected a refusal, got a result", flush=True)
        return
    ident = (
        " inplace-identity=OK"
        if (getattr(pc, "inplace", False) and out_t is tx)
        else (" inplace-identity=BAD" if getattr(pc, "inplace", False) else "")
    )
    out = ttnn.to_torch(out_t)
    exp = torch_rms_norm_ttnn(x, epsilon=1e-12, weight=tw)
    a, e = out.float().flatten(), exp.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    print(f"{lab:60s} {'OK ' if pcc>0.995 else 'BAD'} pcc={pcc:.6f}{ident}", flush=True)


S = ttnn.TensorMemoryLayout
# --- DEFAULT variant, interleaved ---
run("pc default (1,1,384,768) INTL", (1, 1, 384, 768), S.INTERLEAVED, "default")
run("pc default legacy_reduction", (1, 1, 384, 768), S.INTERLEAVED, "default", legacy_reduction=True)
run("pc default legacy_rsqrt", (1, 1, 384, 768), S.INTERLEAVED, "default", legacy_rsqrt=True)
run("pc default (1,1,96,104) INTL non-aligned", (1, 1, 96, 104), S.INTERLEAVED, "default")
# --- SHARDED variant ---
run("pc sharded HEIGHT", (1, 1, 384, 768), S.HEIGHT_SHARDED, "sharded")
run("pc sharded WIDTH", (1, 1, 64, 1536), S.WIDTH_SHARDED, "sharded")
run("pc sharded BLOCK", (1, 1, 384, 768), S.BLOCK_SHARDED, "sharded")
run(
    "pc sharded subblock_w=2 pinned",
    (1, 1, 64, 1536),
    S.WIDTH_SHARDED,
    "sharded",
    shard=([64, 256], (6, 1)),
    subblock_w=2,
)
# --- inplace ---
run("pc inplace HEIGHT", (1, 1, 256, 1024), S.HEIGHT_SHARDED, "sharded", inplace=True)
run("pc inplace WIDTH", (1, 1, 64, 1024), S.WIDTH_SHARDED, "sharded", inplace=True)
run("pc inplace BLOCK", (1, 1, 256, 1024), S.BLOCK_SHARDED, "sharded", inplace=True)
# --- refusals ---
run("refuse use_welford", (1, 1, 384, 768), S.INTERLEAVED, "default", expect_refuse=True, use_welford=True)
run("refuse subblock_w=0", (1, 1, 384, 768), S.HEIGHT_SHARDED, "sharded", expect_refuse=True, subblock_w=0)
run("refuse default-vs-sharded", (1, 1, 384, 768), S.HEIGHT_SHARDED, "default", expect_refuse=True)
run(
    "refuse inplace + DRAM memcfg",
    (1, 1, 384, 768),
    S.HEIGHT_SHARDED,
    "sharded",
    expect_refuse=True,
    mc_override="dram",
    inplace=True,
)
run("refuse block_w restatement", (1, 1, 384, 768), S.HEIGHT_SHARDED, "sharded", expect_refuse=True, block_w=3)
run("refuse subblock_w > dest", (1, 1, 384, 768), S.HEIGHT_SHARDED, "sharded", expect_refuse=True, subblock_w=24)
ttnn.close_device(device)
