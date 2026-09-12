#!/usr/bin/env python3
"""Perf experiment for PR #56146 (tensorbin payload 64B alignment).

Uses REAL Wan2.2 T2V transformer weights, cached to .tensorbin on local disk, then
loaded to a Wormhole/Blackhole mesh with and without the payload alignment fix.

The fix only changes the WRITER (dump_tensor pads the header so the payload starts
64B-aligned). The reader is unchanged. So we generate ALIGNED files with the fixed
dump_tensor, then synthesize UNALIGNED twins by inserting 8 zero bytes before the
payload and bumping header_size by 8 -- identical payload bytes, only the source
pointer offset differs (8 mod 16 on Wormhole / 8 mod 64 on Blackhole => dispatch
rejects the pinned path, exactly like the old unpadded writer).

  aligned   payload_off % 64 == 0   -> pinned host->device DMA (fast)
  unaligned payload_off % 64 == 8   -> rejected -> staged hugepage copy (slow)

IMPORTANT: the pinned path only exists when IOMMU is enabled on the host
(tt_metal/distributed/pinned_memory.cpp:GetMemoryPinningParameters). With IOMMU off
both variants take the staged copy and are indistinguishable. Run `preflight` first.

Subcommands:
  preflight : generate a tiny aligned+unaligned pair, load both, assert the pinned
              path is actually active (unaligned rejects > 0, aligned rejects == 0).
  gen       : read real Wan2.2 weights -> aligned .tensorbin files
  misalign  : aligned files -> unaligned twins (+8 byte header, load-equivalent)
  bench     : load a cache dir to the mesh, time it (rejection lines are counted by
              the runner grepping stderr for "Pinned source memory")
"""
import argparse, glob, json, os, pathlib, sys, time

# --- Wan2.2 T2V transformer weight discovery -------------------------------------
REPO_DIRNAME = "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers"


def find_transformer_dir():
    """Locate the Wan2.2 T2V transformer safetensors dir. Override with WAN_TRANSFORMER_DIR."""
    env = os.environ.get("WAN_TRANSFORMER_DIR")
    if env:
        if not os.path.isfile(os.path.join(env, "diffusion_pytorch_model.safetensors.index.json")):
            sys.exit(f"WAN_TRANSFORMER_DIR={env} has no safetensors index")
        return env
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    home = os.path.expanduser("~")
    roots += [
        os.path.join(home, ".cache/huggingface/hub"),
        "/localdev/*/.cache/huggingface/hub",
        "/root/.cache/huggingface/hub",
    ]
    for root in roots:
        for snaps in glob.glob(os.path.join(root, REPO_DIRNAME, "snapshots", "*")):
            tdir = os.path.join(snaps, "transformer")
            if os.path.isfile(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")):
                return tdir
    sys.exit(
        "Could not find Wan2.2 T2V transformer weights. Set WAN_TRANSFORMER_DIR to the "
        "transformer/ dir that contains diffusion_pytorch_model.safetensors.index.json"
    )


# Big 2D weights per block; each >32MB in bf16 so a replicated per-device upload trips the
# >32MB pinned-H2D threshold and is subject to the NoC-L1 alignment check.
BIG_SUFFIXES = [
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "ffn.net.0.proj.weight",
    "ffn.net.2.weight",
]


def selected_weight_names(n_blocks):
    return [f"blocks.{b}.{s}" for b in range(n_blocks) for s in BIG_SUFFIXES]


def _gen(names, tdir, out):
    import torch, ttnn
    from safetensors import safe_open

    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weight_map = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    manifest, total = [], 0
    t0 = time.time()
    for shard_file, wnames in sorted(by_file.items()):
        with safe_open(os.path.join(tdir, shard_file), framework="pt", device="cpu") as f:
            for n in wnames:
                w = f.get_tensor(n).to(torch.bfloat16)
                tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                fname = out / (n + ".tensorbin")
                ttnn.dump_tensor(str(fname), tt)
                sz = fname.stat().st_size
                total += sz
                with open(fname, "rb") as fh:
                    hs = int.from_bytes(fh.read(8), "little")
                manifest.append(
                    {
                        "name": n,
                        "file": str(fname),
                        "shape": list(w.shape),
                        "bytes": sz,
                        "header_size": hs,
                        "payload_off": 8 + hs,
                        "payload_mod64": (8 + hs) % 64,
                    }
                )
                del w, tt
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"[gen] wrote {len(manifest)} tensors, {total/1e9:.2f} GB in {time.time()-t0:.1f}s -> {out}")
    print(f"[gen] payload_offset mod 64 values: {sorted(set(m['payload_mod64'] for m in manifest))} (0 == aligned)")
    return manifest


def cmd_gen(args):
    _gen(selected_weight_names(args.blocks), find_transformer_dir(), args.out)


def _misalign(src, dst):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(src / "manifest.json"))
    ZERO8 = b"\x00" * 8
    new = []
    t0 = time.time()
    for m in manifest:
        sp = pathlib.Path(m["file"])
        dp = dst / sp.name
        data = open(sp, "rb").read()
        hs = int.from_bytes(data[:8], "little")
        header, payload = data[8 : 8 + hs], data[8 + hs :]
        new_hs = hs + 8  # keeps payload 8B-aligned but 8 mod 16 -> rejected
        with open(dp, "wb") as f:
            f.write(new_hs.to_bytes(8, "little"))
            f.write(header)
            f.write(ZERO8)
            f.write(payload)
        nm = dict(m)
        nm.update(file=str(dp), header_size=new_hs, payload_off=8 + new_hs, payload_mod64=(8 + new_hs) % 64)
        new.append(nm)
    json.dump(new, open(dst / "manifest.json", "w"), indent=2)
    print(f"[misalign] wrote {len(new)} twins in {time.time()-t0:.1f}s -> {dst}")
    print(
        f"[misalign] payload_off mod 64: {sorted(set(m['payload_mod64'] for m in new))}; "
        f"mod 16: {sorted(set(m['payload_off']%16 for m in new))} (nonzero == rejected on Wormhole)"
    )
    return new


def cmd_misalign(args):
    _misalign(args.src, args.dst)


def _open_mesh(devices):
    import ttnn

    try:
        n_avail = ttnn.get_num_devices()
    except Exception:
        n_avail = devices
    devices = min(devices, n_avail) if n_avail else devices
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, devices))
    return mesh, devices


def _bench(cache, devices, limit=0):
    import ttnn

    cache = pathlib.Path(cache)
    manifest = json.load(open(cache / "manifest.json"))
    if limit:
        manifest = manifest[:limit]
    mesh, devices = _open_mesh(devices)
    print(f"[bench] opened mesh 1x{devices}; loading {len(manifest)} tensors from {cache}", flush=True)
    warm = ttnn.load_tensor(manifest[0]["file"], device=mesh)  # exclude setup from timing
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(warm)
    per = []
    t0 = time.time()
    for m in manifest:
        s = time.time()
        t = ttnn.load_tensor(m["file"], device=mesh)
        ttnn.synchronize_device(mesh)
        per.append(time.time() - s)
        ttnn.deallocate(t)
    total = time.time() - t0
    ttnn.close_mesh_device(mesh)
    gb = sum(m["bytes"] for m in manifest) / 1e9
    print(
        f"[bench] RESULT cache={cache.name} tensors={len(manifest)} devices={devices} "
        f"total_load_time={total:.3f}s file_bytes={gb:.2f}GB "
        f"mean_ms={sum(per)/len(per)*1000:.2f} max_ms={max(per)*1000:.2f}"
    )


def cmd_bench(args):
    _bench(args.cache, args.devices, args.limit)


def cmd_preflight(args):
    """Prove the pinned path is active: 1 block, aligned vs unaligned, count rejects via a child grep.
    Because rejects are logged from C++, we re-exec bench in a subprocess and grep its stderr."""
    import subprocess

    work = pathlib.Path(args.work)
    al, un = work / "pf_aligned", work / "pf_unaligned"
    _gen(selected_weight_names(1), find_transformer_dir(), al)
    _misalign(al, un)
    env = dict(os.environ)

    def run_and_count(cache):
        p = subprocess.run(
            [sys.executable, __file__, "bench", "--cache", str(cache), "--devices", str(args.devices)],
            capture_output=True,
            text=True,
            env=env,
        )
        log = p.stdout + p.stderr
        rej = log.count("Pinned source memory")
        res = next((ln for ln in log.splitlines() if "RESULT" in ln), "(no RESULT)")
        return rej, res, log

    ar, ares, alog = run_and_count(al)
    ur, ures, ulog = run_and_count(un)
    print("\n==================== PREFLIGHT ====================")
    print(f"aligned  : rejects={ar}  {ares}")
    print(f"unaligned: rejects={ur}  {ures}")
    if ur > 0 and ar == 0:
        print("PREFLIGHT PASS: pinned path is active; unaligned files are rejected. Safe to run full bench.")
        return 0
    print("PREFLIGHT FAIL: unaligned produced no rejects => pinned path NOT active on this host.")
    print("  Most likely IOMMU is off. Check: `ls /sys/kernel/iommu_groups | wc -l` (should be > 0)")
    print("  and that /proc/cmdline contains intel_iommu=on. The perf delta cannot be measured without it.")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--blocks", type=int, default=40)
    mi = sub.add_parser("misalign")
    mi.add_argument("--src", required=True)
    mi.add_argument("--dst", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--cache", required=True)
    b.add_argument("--devices", type=int, default=4)
    b.add_argument("--limit", type=int, default=0)
    pf = sub.add_parser("preflight")
    pf.add_argument("--work", required=True)
    pf.add_argument("--devices", type=int, default=4)
    args = ap.parse_args()
    {"gen": cmd_gen, "misalign": cmd_misalign, "bench": cmd_bench, "preflight": cmd_preflight}[args.cmd](args)


if __name__ == "__main__":
    main()
