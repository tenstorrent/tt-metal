# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a HOST-SPECIFIC PP=4 x (8,1) rank binding from the live galaxy.

Why this exists: the [8,1] column -> physical device map is a per-galaxy property, and a wrong map
does NOT error. It silently builds a 4-stage pipeline whose stages are not galaxy columns, and the
perf numbers come out quietly wrong. Measured on two 32-chip Blackhole galaxies:

    rank  bh-glx-b03u02              bh-glx-110-a04u02
    0     0,1,2,3,11,10,9,8          0,1,2,3,27,26,25,24
    1     4,5,6,7,15,14,13,12        4,5,6,7,31,30,29,28
    2     28,29,30,31,23,22,21,20    12,13,14,15,23,22,21,20
    3     24,25,26,27,19,18,17,16    8,9,10,11,19,18,17,16

Every rank differs. So derive it instead of hardcoding: open the 8x4 mesh with the torus_xy
descriptor and ask create_submeshes(MeshShape(8,1)) for each column's device ids -- the identical
carving the single-process test uses.

--layout 4x4 carves the same 8x4 into create_submeshes(MeshShape(4,4)) instead: the top tray half
(rows 0-3) and the bottom one (rows 4-7), the stages of the 4x4 SC4 pipeline. The bottom half's rows
are listed in reverse, asking for TP row t on physical row 7-t (shorter top/bottom D2D hops). The
topology mapper does not honour list order today -- it places the half's rows 4..7 in natural
order -- so taking effect needs mesh-graph-descriptor pinnings (ci/sc4_4x4_sp1_mgd.textproto
carries them, Rev C only). For more than one galaxy, stitch the per-host outputs with
assemble_pipeline_binding.py.

Usage (galaxy must be idle):
    python gen_pipeline_binding.py [--layout 8x1|4x4] [--template <yaml>] [--out <yaml>] [--profile]

Writes <template stem>.<hostname>.yaml next to the template by default and prints the path.
"""
import argparse
import os
import re
import socket
import sys
from pathlib import Path

from loguru import logger

TOPO = Path("models/demos/common/prefill/runners/topology_configuration")
DEFAULT_TEMPLATE = TOPO / "pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml"
# --profile has its OWN template, and therefore its own output name. Sharing the plain template
# would write both variants to <stem>.<host>.yaml and the second run would silently overwrite the
# first -- leaving an "e2e" binding that carries TT_METAL_PROFILER_DIR, so every matrix cell runs
# instrumented and reports inflated times with no indication why.
PROFILE_TEMPLATE = TOPO / "pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y_profile.yaml"
TEMPLATE_4X4 = TOPO / "pipeline_prefill_request_intragalaxy_2rank_4x4_sp1.yaml"

# layout -> (submesh shape, number of submeshes)
LAYOUTS = {"8x1": ((8, 1), 4), "4x4": ((4, 4), 2)}


def probe_submeshes(shape, count):
    import ttnn
    from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        create_fabric_router_config(max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
    )
    md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    try:
        full = list(md.get_device_ids())
        subs = [list(sm.get_device_ids()) for sm in md.create_submeshes(ttnn.MeshShape(*shape))]
    finally:
        ttnn.close_mesh_device(md)
    size = shape[0] * shape[1]
    if len(subs) != count or any(len(s) != size for s in subs):
        raise SystemExit(f"expected {count} submeshes of {size}, got {[len(s) for s in subs]}")
    return full, subs


def mirror_rows(ids, shape):
    """Reverse the row order of a row-major device list."""
    rows, cols = shape
    return [d for r in reversed(range(rows)) for d in ids[r * cols : (r + 1) * cols]]


def fill_rank_bindings(text, subs, tpl, profile_root=None):
    """Write subs[r] into rank r's TT_VISIBLE_DEVICES in a rank-binding template's text.

    Each rank must have exactly one `{rank: r, mesh_id: r, ...}` line; assemble_pipeline_binding.py
    fills the multi-host template through this too."""
    for r, sub in enumerate(subs):
        ids = ",".join(map(str, sub))
        extra = f'TT_METAL_PROFILER_DIR: "{profile_root}/rank{r}", ' if profile_root else ""
        pat = re.compile(
            r"(\{rank: %d, mesh_id: %d, mesh_host_rank: 0, env_overrides: \{)(?:TT_METAL_PROFILER_DIR: \"[^\"]*\", )?TT_VISIBLE_DEVICES: \"[^\"]*\"\}\}"
            % (r, r)
        )
        text, n = pat.subn(lambda m: f'{m.group(1)}{extra}TT_VISIBLE_DEVICES: "{ids}"}}}}', text)
        if n != 1:
            raise SystemExit(
                f"template does not have exactly one rank {r} binding line (found {n}); is {tpl} the right shape?"
            )
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="8x1", help="stage shape to carve the 8x4 into")
    # Default resolved after parsing, because it depends on --layout and --profile.
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", action="store_true", help="also add a per-rank TT_METAL_PROFILER_DIR")
    ap.add_argument("--profile-root", default="M4_PROFILE_OUT")
    args = ap.parse_args()

    host = socket.gethostname()
    # An explicit --template always wins; otherwise --profile picks the profile template so the two
    # variants land on different output names (..._torus_y.<host>.yaml vs ..._torus_y_profile.<host>.yaml),
    # which is what the reference branch's run_matrix.sh / run_single_layer_profile.sh each pick up.
    if args.layout == "4x4" and args.profile and not args.template:
        raise SystemExit("--layout 4x4 has no profile template; pass --template")
    if args.template:
        tpl = Path(args.template)
    elif args.layout == "4x4":
        tpl = TEMPLATE_4X4
    else:
        tpl = PROFILE_TEMPLATE if args.profile else DEFAULT_TEMPLATE
    out = Path(args.out) if args.out else tpl.with_name(f"{tpl.stem}.{host}.yaml")

    shape, count = LAYOUTS[args.layout]
    full, subs = probe_submeshes(shape, count)
    if args.layout == "4x4":
        subs[1] = mirror_rows(subs[1], shape)
    logger.info(f"full 8x4 device ids (row-major): {full}")
    for i, s in enumerate(subs):
        logger.info(f"  rank{i}: {','.join(map(str, s))}")

    text = fill_rank_bindings(tpl.read_text(), subs, tpl, profile_root=args.profile_root if args.profile else None)

    banner = (
        f"# GENERATED by gen_pipeline_binding.py on {host} -- do not hand-edit the TT_VISIBLE_DEVICES lists.\n"
        f"# The [{shape[0]},{shape[1]}] submesh -> device map is per-galaxy; regenerate on every new machine.\n"
    )
    out.write_text(banner + text)
    logger.info(f"wrote {out} ({len(subs)} rank bindings)")


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))
    main()
