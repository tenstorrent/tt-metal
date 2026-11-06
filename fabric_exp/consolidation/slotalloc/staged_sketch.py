import math
# Full-rate depths (ALLOCATOR_NEEDS_FIRST.md section 3; fabric_slot_allocator.hpp's *_full_rate_depth_* constants)
FULL_RATE_DEPTH = {('vc0','w','2d'):18,('vc0','f','2d'):16,('vc0','r','2d'):16,
        ('vc0','w','1d'):24,('vc0','f','1d'):24,('vc0','r','1d'):32,
        ('vc1','f','2d'):10,('vc1','r','2d'):8,('vc2','w','2d'):4,('vc2','r','2d'):4}
MIN, CAP = 2, 127
RX_LEFTOVER_CAP = 32  # Limits::max_receiver_slots_from_leftovers: the deepest receiver measured
def pow2_nearest(x):
    if x <= 2: return 2
    lo = 2 ** math.floor(math.log2(x)); hi = lo * 2
    return hi if (hi - x) <= (x - lo) else lo
def channels(cfg):
    dim = '1d' if cfg['1d'] else '2d'; nf = cfg.get('nf', 1 if cfg['1d'] else 3)  # nf, nf1: VC0 / VC1 forwarding channels
    ch = [('vc0','w',dim,0)] + [('vc0','f',dim,i) for i in range(nf)] + [('vc0','r',dim,0)]
    if cfg.get('vc1'): ch += [('vc1','f','2d',i) for i in range(cfg.get('nf1', 3))] + [('vc1','r','2d',0)]
    if cfg.get('vc2'): ch += [('vc2','w','2d',0), ('vc2','r','2d',0)]
    return ch
def full_rate_depth(c): return FULL_RATE_DEPTH[c[:3]]
def allocate(cfg, N, leftover):
    # The port's fabric_slot_allocator.cpp, stage for stage. A VC's forwarding channels are one unit with one count
    # (the router keeps one downstream forwarding ring size per VC); each worker is a unit of its own.
    chs = channels(cfg); rx = [c for c in chs if c[1]=='r']; tx = [c for c in chs if c[1]!='r']
    total = sum(full_rate_depth(c) for c in chs)
    f = min(1.0, N / total)
    out = {c: pow2_nearest(f * full_rate_depth(c)) for c in rx}            # stage 1: receivers, nearest pow2
    B = N - sum(out.values())
    units = []                                                             # lists of channels sharing one count
    for c in tx:
        u = next((u for u in units if c[1] == 'f' and u[0][1] == 'f' and u[0][0] == c[0]), None)
        if u is None: units.append([c])
        else: u.append(c)
    sdep = sum(full_rate_depth(c) for c in tx)
    fs = min(1.0, B / sdep)
    cnt = [0] * len(units)
    if B >= sdep:                                                          # stage 2: every depth fits
        cnt = [min(full_rate_depth(u[0]), CAP) for u in units]
    else:                                                                  # stage 2: squeeze, integer shares
        d = [full_rate_depth(u[0]) for u in units]
        flo = [x * B // sdep for x in d]; rem = [x * B % sdep for x in d]
        cnt = [max(x, MIN) for x in flo]
        given = sum(cnt[k] * len(units[k]) for k in range(len(units)))
        vcn = lambda u: int(u[0][0][2])  # noqa: E731
        order = sorted(range(len(units)), key=lambda k: (-rem[k], vcn(units[k]), 0 if units[k][0][1] == 'f' else 1, units[k][0][3]))
        for k in order:
            if cnt[k] == flo[k] and flo[k] >= MIN and given + len(units[k]) <= B:
                cnt[k] += 1; given += len(units[k])
        while given > B:
            k = max((k for k in range(len(units)) if cnt[k] > MIN), key=lambda k: cnt[k] - MIN, default=None)
            if k is None: raise ValueError("the senders' minimums do not fit")
            cnt[k] -= 1; given -= len(units[k])
    for k, u in enumerate(units):
        for c in u: out[c] = cnt[k]
    L = N - sum(out.values())
    for group in leftover:                                                 # stage 3: whole steps, groups in order
        r = [c for c in rx if (c[0], c[1]) in group]                       # receivers: each to its next power of 2
        step = sum(out[c] for c in r)
        if r and step <= L and all(2 * out[c] <= min(CAP, RX_LEFTOVER_CAP) for c in r):
            for c in r: out[c] *= 2
            L -= step
        g = [u for k in group for u in units if (u[0][0], u[0][1]) == k]  # senders: units in the group's order,
        grew = bool(g)                                                     # +1 on each unit in turn while it fits
        while grew:
            grew = False
            for u in g:
                if len(u) <= L and out[u[0]] < CAP:
                    for c in u: out[c] += 1
                    L -= len(u); grew = True
    return out, f, fs, L
def fmt(cfg, out):
    def vc(v):
        w = [out[c] for c in out if c[0]==v and c[1]=='w']; fw = [out[c] for c in out if c[0]==v and c[1]=='f']; r = [out[c] for c in out if c[0]==v and c[1]=='r']
        fs = '/'.join(map(str, fw)) if len(set(fw)) > 1 else (str(fw[0]) if fw else '')
        return ','.join(x for x in [str(w[0]) if w else '', fs, str(r[0]) if r else ''] if x)
    s = vc('vc0')
    if cfg.get('vc1'): s += ' + ' + vc('vc1')
    if cfg.get('vc2'): s += ' + ' + vc('vc2')
    return s
CFGS = [('2D mesh/torus', {'1d':False}), ('1D line/ring', {'1d':True}), ('1D neighbor exchange', {'1d':True,'nf':0}), ('VC1 split', {'1d':False,'vc1':True}),
        ('VC2 on', {'1d':False,'vc2':True}), ('VC1 + VC2', {'1d':False,'vc1':True,'vc2':True})]
NS = {4352:(81,83), 3072:(115,117), 8704:(41,41), 15232:(23,23)}
LEFT_2D = [[('vc0','r')], [('vc0','f')], [('vc0','w')]]   # default_leftover_order(one_d = false)
LEFT_1D = [[('vc0','r')], [('vc0','w'), ('vc0','f')]]      # default_leftover_order(one_d = true)
def leftover(cfg): return LEFT_1D if cfg['1d'] else LEFT_2D
print('| config | ' + ' | '.join(f'{s} B (N {NS[s][0]}{"/"+str(NS[s][1]) if NS[s][1]!=NS[s][0] else ""})' for s in NS) + ' |')
print('| --- |' + ' --- |' * len(NS))
for name, cfg in CFGS:
    cells = []
    for s, (n2, n1) in NS.items():
        N = n1 if cfg['1d'] else n2
        try:
            out, f, fs, L = allocate(cfg, N, leftover(cfg))
            assert sum(out.values()) + L == N
            cells.append(fmt(cfg, out) + (f' (f {f:.2f})' if f < 1 else ''))
        except Exception as e: cells.append('err ' + str(e))
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
