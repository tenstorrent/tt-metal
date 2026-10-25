import json, re, collections
DB = "/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/cmds.json"
d = json.load(open(DB))["O3"]
vc1 = set(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt").read().split())
def ctmap(cmd):
    cache = re.search(r"(\S+/kernels/fabric_erisc_router_v2/\d+)/", cmd).group(1)
    s = open(cache + "/named_ct_arg_map_generated.h").read()
    return dict((k, int(v)) for k, v in re.findall(r'\{"(\w+)",(\d+)\}', s))
rows = collections.defaultdict(list)
for h, procs in d.items():
    c = procs.get("subordinate_active_erisc", {}).get("compile")
    if not c: continue
    m = ctmap(c); D = set(re.findall(r"-D(\w+)", c))
    if m["NUM_ACTIVE_ERISCS"] != 2: continue
    key = ("2D" if "FABRIC_2D" in D else "1D", "VC1" if h in vc1 else "", "dir%d" % m["MY_DIRECTION"], "ds%d" % m["VC0_DOWNSTREAM_EDM_SIZE"],
           "edge%d%d%d" % (m["IS_INTERMESH_ROUTER"], m["IS_INTERMESH_ROUTER_ON_EDGE"], m["IS_INTRAMESH_ROUTER_ON_EDGE"]), "dla%d" % m["ENABLE_DEADLOCK_AVOIDANCE"], "spd%d" % m["ENABLE_SPEEDY_VC0"])
    rows[key].append(h)
for k, v in sorted(rows.items()):
    print(len(v), *k, v[:3])
