import sys,subprocess,os
T=os.path.dirname(os.path.abspath(__file__))
def rid_of(a): return subprocess.run(["python3",f"{T}/profparse.py",f"{T}/csv/{a}.csv"],capture_output=True,text=True).stdout.strip().splitlines()[-1].split()
def z(a,rid,risc):
    out=subprocess.run(["python3",f"{T}/zones.py",f"{T}/csv/{a}.csv",rid],capture_output=True,text=True,env=dict(os.environ,ZRISC=risc)).stdout
    d={}
    for l in out.splitlines():
        p=l.split()
        if len(p)>12 and p[1].startswith("#"): d.setdefault((p[0],int(p[1][1:-1])),[]).append(dict(smed=float(p[4]),emed=float(p[7]),emax=float(p[9]),dmed=float(p[12]),dmax=float(p[14])))
    return d
for a in sys.argv[1:]:
    pp=rid_of(a); rid=pp[1]; kus=float(pp[4])
    t0=z(a,rid,"TRISC_0"); b=z(a,rid,"BRISC"); nc=z(a,rid,"NCRISC")
    tails=[]; mcs=[]
    for L in (1,2,3):
        gu=t0[("C-GU",L)][0]; dn=t0[("C-DN",L)][0]; tails.append(dn['emed']-gu['emax'])
        for d in (b,nc):
            for v in d.get(("ROOT-MC",L),[]): mcs.append(v['dmed'])
    print(f"{a:8s} kernel {kus:7.1f} us | tail GUmax->DNend L1/L2/L3 {' / '.join(f'{x:5.1f}' for x in tails)} | ROOT-MC dur med (per root,L1-3) avg {sum(mcs)/len(mcs):.1f}")
