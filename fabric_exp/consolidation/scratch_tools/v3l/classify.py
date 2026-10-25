import json,re,collections,os,sys
db=json.load(open('/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/cmds.json'))
vc1=set(open('/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt').read().split())
def info(opt='O3'):
    out={}
    for h,v in db[opt].items():
        for proc,c in v.items():
            cmd=c['compile']
            d=re.search(r"(\S+/kernels/fabric_erisc_router_v2/"+h+")/",cmd).group(1)
            m=dict((k,int(x)) for k,x in re.findall(r'\{"(\w+)",(-?\d+)\}',open(d+'/named_ct_arg_map_generated.h').read()))
            defs=re.findall(r'-D(\w+)',cmd)
            m['_2D']='FABRIC_2D' in defs
            m['_VC1ACT']='FABRIC_2D_VC1_ACTIVE' in defs
            m['_VC1SRV']='FABRIC_2D_VC1_SERVICED' in defs
            m['_proc']=proc
            m['_vc1']=h in vc1
            m['_defs']=[x for x in defs if 'FABRIC' in x]
            out[(h,proc)]=m
    return out
if __name__=='__main__':
    inf=info()
    cnt=collections.Counter()
    for (h,p),m in inf.items():
        key=(p,m['MY_ERISC_ID'],m['NUM_ACTIVE_ERISCS'],'2D' if m['_2D'] else '1D','vc1' if m['_vc1'] else '', 'speedy' if m['ENABLE_SPEEDY_VC0'] else '', 'DA' if m['ENABLE_DEADLOCK_AVOIDANCE'] else '','S'+''.join(str(m['IS_SENDER_CHANNEL_%d_SERVICED'%i]) for i in range(10)),'R'+''.join(str(m['IS_RECEIVER_CHANNEL_%d_SERVICED'%i]) for i in range(3)))
        cnt[key]+=1
    for k,v in sorted(cnt.items()): print(v,k)
