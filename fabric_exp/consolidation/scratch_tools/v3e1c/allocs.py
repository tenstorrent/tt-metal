import json,re,sys,collections
d=json.load(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/cmds.json"))["O3"]
vc1=set(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt").read().split())
def ctmap(cmd):
    cache = re.search(r"(\S+/kernels/fabric_erisc_router_v2/\d+)/", cmd).group(1)
    s = open(cache + "/named_ct_arg_map_generated.h").read()
    return dict((k, int(v)) for k, v in re.findall(r'\{"(\w+)",(\d+)\}', s))
def parse(args, base, ns, nr):
    assert args[base]==0xabcd1234, hex(args[base])
    n=args[base+1]; ents=[tuple(args[base+2+4*i:base+6+4*i]) for i in range(n)]
    mb=base+2+4*n
    snd=args[mb:mb+ns]; rcv=args[mb+ns:mb+ns+nr]
    return ents,snd,rcv,mb+ns+nr
res=collections.Counter()
for h,procs in d.items():
    for p,cc in procs.items():
        c=cc.get("compile")
        if not c: continue
        m=ctmap(c)
        if m.get("NUM_ACTIVE_ERISCS")!=2: continue
        a=[int(x) for x in re.search(r"-DKERNEL_COMPILE_TIME_ARGS=(\S+)", c).group(1).split(",")]
        ns,nr=m["NUM_SENDER_CHANNELS"],m["NUM_RECEIVER_CHANNELS"]
        ents,snd,rcv,end=parse(a,0,ns,nr)
        rents,_,rrcv,_=parse(a,end+1,0,nr)
        loc=[ents[i][:2] for i in rcv]; rem=[rents[i][:2] for i in rrcv]
        D=set(re.findall(r"-D(FABRIC_\w+)",c))
        tag=("VC1" if h in vc1 else "VC0")+(" xover" if "FABRIC_2D_VC0_CROSSOVER_TO_VC1" in D else "")+" "+p[:3]
        res[(tag, "same" if loc==rem else "DIFF %s vs %s"%(loc,rem))]+=1
for k,v in sorted(res.items()): print(v,k)
