import sys,subprocess,glob,os,re
RE="/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf"
vc1=set(open('/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt').read().split())
def loadsz(elf):
    out=subprocess.run([RE,'-lW',elf],capture_output=True,text=True).stdout
    return sum(int(l.split()[4],16) for l in out.splitlines() if l.strip().startswith('LOAD'))
for d in sys.argv[1:]:
    e0=[];e1=[]
    for h in vc1:
        for p,lst in (('active_erisc',e0),('subordinate_active_erisc',e1)):
            f=f'{d}/{h}/{p}/{p}.elf'
            if os.path.exists(f): lst.append((loadsz(f),h))
    m0=max(e0) if e0 else (0,''); m1=max(e1) if e1 else (0,'')
    print(f'{d}: E0 max {m0[0]} ({len(e0)} cfgs) E1 max {m1[0]} ({len(e1)}) sum {m0[0]+m1[0]}')
