# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Rev C slicing corrector for the single-pod pipeline bootstrap.

Called from bootstrap_pipeline_dir.sh after the upstream slicer has produced
slice_to_pcie_device_mapping.yaml. Reads its inputs from the environment:

    PIPELINE_DIR     bundle directory written by the slicer
    PIPELINE_CONFIG  scaleout config (used for stage/slice/host placeholders)
    HOSTFILE         hostfile passed to the slicer
    TRAY_DIR         dir containing per-host tray_to_pcie_device_mapping yamls
    EXPECTED_RB      path to the rank-binding yaml to rewrite

Rewrites slice_to_pcie_device_mapping.yaml and the rank-binding yaml in place
so each rank's TT_VISIBLE_DEVICES matches the Rev C tray-swap layout.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PIPELINE_DIR = Path(os.environ["PIPELINE_DIR"])
PIPELINE_CONFIG = Path(os.environ["PIPELINE_CONFIG"])
HOSTFILE = Path(os.environ["HOSTFILE"])
TRAY_DIR = Path(os.environ["TRAY_DIR"])
RB_PATH = Path(os.environ["EXPECTED_RB"])
SLICE_PATH = PIPELINE_DIR / "slice_to_pcie_device_mapping.yaml"


def load_tray_map(yaml_text):
    """Load `tray_to_pcie_device_mapping.yaml` -> {tray_id: {logical_ids}}.

    The C++ test writes a doc shaped like:
      device_mapping:
        1: [<logical_id>, <logical_id>, ...]
        2: [...]
        3: [...]
        4: [...]
      arch: BLACKHOLE
    """
    doc = yaml.safe_load(yaml_text)
    if not isinstance(doc, dict) or "device_mapping" not in doc:
        sys.exit("[rev_c] tray map yaml missing 'device_mapping' top-level key")
    out = {}
    for tray_id, ids in doc["device_mapping"].items():
        out[int(tray_id)] = set(int(x) for x in ids)
    return out


# Step 1: collect & sanity-check per-host tray maps
tray_maps = {}
for f in sorted(TRAY_DIR.glob("*.yaml")):
    host = f.stem
    tm = load_tray_map(f.read_text())
    if set(tm) != {1, 2, 3, 4}:
        sys.exit(f"[rev_c] {f.name}: expected trays {{1,2,3,4}}, got {sorted(tm)}")
    for t, ids in tm.items():
        if len(ids) != 8:
            sys.exit(f"[rev_c] {f.name}: tray {t}: expected 8 chip ids, got {len(ids)} ({sorted(ids)})")
    tray_maps[host] = tm

# Step 2: load slicer output, sanity-check that its chip ids align per-host
# with the discovery-binary tray map (catches anything else that might drive
# the two sources out of sync before the correction produces a wrong yaml).
with open(SLICE_PATH) as fp:
    slice_doc = yaml.safe_load(fp)
device_mapping = slice_doc["device_mapping"]
for host, slices in device_mapping.items():
    if host not in tray_maps:
        sys.exit(f"[rev_c] host {host} in slice yaml but no tray map collected for it")
    sliced = set()
    for ids in slices.values():
        sliced.update(ids)
    trayed = set().union(*tray_maps[host].values())
    if sliced != trayed:
        sys.exit(
            f"[rev_c] {host}: slice yaml chip ids {sorted(sliced)} != "
            f"tray map chip ids {sorted(trayed)}; "
            "the slicer and discovery binary disagree on this host's chip set."
        )


# Step 3: apply the four-equation Rev C correction per host.
def correct_for_rev_c(slices, trays):
    s = {sid: set(ids) for sid, ids in slices.items()}
    new = {
        0: (s[0] & trays[1]) | (s[3] & trays[2]),
        1: (s[1] & trays[1]) | (s[2] & trays[2]),
        2: (s[1] & trays[3]) | (s[2] & trays[4]),
        3: (s[0] & trays[3]) | (s[3] & trays[4]),
    }
    for sid, ids in new.items():
        if len(ids) != 8:
            raise SystemExit(
                f"[rev_c] post-correction sanity check failed: slice {sid} "
                f"has {len(ids)} chips (expected 8): {sorted(ids)}"
            )
    return new


corrected_mapping = {}
for host, slices in device_mapping.items():
    new_slices = correct_for_rev_c(slices, tray_maps[host])
    corrected_mapping[host] = {sid: sorted(ids) for sid, ids in new_slices.items()}
slice_doc["device_mapping"] = corrected_mapping

with open(SLICE_PATH, "w") as fp:
    yaml.dump(slice_doc, fp, default_flow_style=False, sort_keys=False)
print(f"[rev_c]   wrote corrected {SLICE_PATH}")


# Step 4: regenerate rank-binding yaml. Mirror the upstream
# generate_blitz_decode_pipeline_configs.py logic: load pipeline_config,
# remap placeholder host-N -> real host via canonical sort of hostfile,
# then write per-rank TT_VISIBLE_DEVICES from corrected slice data.
def sort_hosts_canonical(hosts):
    parsed = []
    for h in hosts:
        m = re.search(r"(\d+)u(\d{2})$", h)
        if not m:
            sys.exit(f"[rev_c] hostname {h!r} doesn't match '<digits>u<2 digits>' canonical pattern")
        parsed.append((h, int(m.group(1)), int(m.group(2))))
    groups = defaultdict(list)
    for h, host_num, u_num in parsed:
        groups[host_num].append((u_num, h))
    out = []
    for idx, host_num in enumerate(sorted(groups)):
        groups[host_num].sort(key=lambda e: e[0], reverse=(idx % 2 == 0))
        out.extend(h for _, h in groups[host_num])
    return out


with open(PIPELINE_CONFIG) as fp:
    pcfg = yaml.safe_load(fp)
with open(HOSTFILE) as fp:
    allocated = sort_hosts_canonical([l.strip() for l in fp if l.strip()])

config_hosts = []
seen = set()
for entry in pcfg["stage_to_slice_mapping"].values():
    if entry["host"] not in seen:
        config_hosts.append(entry["host"])
        seen.add(entry["host"])
if len(config_hosts) != len(allocated):
    sys.exit(
        f"[rev_c] hostfile has {len(allocated)} hosts but pipeline config "
        f"has {len(config_hosts)} unique placeholder hosts"
    )
host_remap = dict(zip(config_hosts, allocated))

rank_bindings = []
for rank in sorted(pcfg["stage_to_slice_mapping"]):
    info = pcfg["stage_to_slice_mapping"][rank]
    real_host = host_remap[info["host"]]
    devices = corrected_mapping[real_host][info["slice"]]
    rank_bindings.append(
        {
            "rank": rank,
            "mesh_id": rank,
            "mesh_host_rank": 0,
            "env_overrides": {"TT_VISIBLE_DEVICES": ",".join(str(d) for d in devices)},
        }
    )

rb_doc = {
    "rank_bindings": rank_bindings,
    "mesh_graph_desc_path": pcfg["mesh_graph_desc_path"],
}
with open(RB_PATH, "w") as fp:
    yaml.dump(rb_doc, fp, default_flow_style=False, sort_keys=False)
print(f"[rev_c]   wrote corrected {RB_PATH}")
