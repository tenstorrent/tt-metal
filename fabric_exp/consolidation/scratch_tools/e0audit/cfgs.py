#!/usr/bin/env python3
"""cfgs.py: classify the logged E0 (active_erisc) configs by their named compile-time args."""
import glob, json, re, os, collections
db = json.load(open('/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/cmds.json'))
vc1 = set(open('/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt').read().split())
hdr = {}
for p in glob.glob('/home/aagarwal/.tt-metal-cache*/*/kernels/fabric_erisc_router_v2/*/named_ct_arg_map_generated.h'):
    hdr.setdefault(p.split('/')[-2], p)
out = {}
for h, v in db['O3'].items():
    if 'active_erisc' not in v:
        continue
    p = hdr.get(h)
    if not p:
        out[h] = None; continue
    a = dict((k, int(x)) for k, x in re.findall(r'\{"(\w+)",(-?\d+)\}', open(p).read()))
    out[h] = a
json.dump(out, open('/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/e0audit/e0cfg.json', 'w'))
c = collections.Counter()
for h, a in out.items():
    if a is None:
        c['nohdr'] += 1; continue
    key = (a['IS_2D_FABRIC'], a['MY_DIRECTION'], a['ENABLE_DEADLOCK_AVOIDANCE'], a['ENABLE_SPEEDY_VC0'] if 'ENABLE_SPEEDY_VC0' in a else -1, a['ACTUAL_VC1_SENDER_CHANNELS'], a['NUM_ACTIVE_ERISCS'], a['ENABLE_CONTEXT_SWITCH'], a['IDLE_CONTEXT_SWITCHING'], a['SWITCH_INTERVAL'], h in vc1)
    c[key] += 1
print('2D dir DA speedy vc1ch nerisc ctxsw idle swint isvc1: count')
for k, n in sorted(c.items(), key=str):
    print(k, n)
