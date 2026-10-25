import json, re, sys, collections, glob, os
DB = "/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/cmds.json"
d = json.load(open(DB))
vc1 = set(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt").read().split())
def ctmap(cmd):
    cache = re.search(r"(\S+/kernels/fabric_erisc_router_v2/\d+)/", cmd).group(1)
    s = open(cache + "/named_ct_arg_map_generated.h").read()
    return dict((k, int(v)) for k, v in re.findall(r'\{"(\w+)",(\d+)\}', s))
def defs(cmd):
    return set(re.findall(r"-D(\w+)", cmd))
rows = {}
for opt in ("O3",):
    for h, procs in d[opt].items():
        for p, c in procs.items():
            if "compile" not in c: continue
            m = ctmap(c["compile"]); D = defs(c["compile"])
            key = (p, m["NUM_ACTIVE_ERISCS"], m["MY_ERISC_ID"], "2D" if "FABRIC_2D" in D else "1D",
                   "VC1" if "FABRIC_2D_VC1_ACTIVE" in D else "", "VC2" if "FABRIC_2D_VC2_SERVICED" in D else "",
                   "speedy" if m["ENABLE_SPEEDY_VC0"] else "", "dla" if m["ENABLE_DEADLOCK_AVOIDANCE"] else "",
                   "fla" if m["ENABLE_FIRST_LEVEL_ACK_VC0"] else "",
                   "rx" + "".join(str(m["IS_RECEIVER_CHANNEL_%d_SERVICED"%i]) for i in range(3)),
                   "tx" + "".join(str(m["IS_SENDER_CHANNEL_%d_SERVICED"%i]) for i in range(10)),
                   "dir%d" % m["MY_DIRECTION"], "edge%d%d%d" % (m["IS_INTERMESH_ROUTER"], m["IS_INTERMESH_ROUTER_ON_EDGE"], m["IS_INTRAMESH_ROUTER_ON_EDGE"]),
                   "same%d" % m["FORCE_ALL_PATHS_TO_USE_SAME_NOC"], "udm%d" % m["UDM_MODE"], "mux%d"%m["FABRIC_TENSIX_EXTENSION_MUX_MODE"],
                   "dis%d%d" % (m["DISABLE_RX_CH0_FORWARDING"], m["DISABLE_RX_CH1_FORWARDING"]),
                   "vc0ds%d" % m["VC0_DOWNSTREAM_EDM_SIZE"], "vc1ds%d" % m["VC1_DOWNSTREAM_EDM_SIZE"],
                   "spinrx%d" % m["ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK"], "vc1h" if h in vc1 else "")
            rows.setdefault(key, []).append(h)
for k, v in sorted(rows.items(), key=lambda kv: (kv[0][0], str(kv[0]))):
    print(len(v), " ".join(str(x) for x in k if x != ""), v[0])
