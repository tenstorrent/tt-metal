#!/usr/bin/env python3
"""Host-only regression cases for actual phase ownership, not just total time."""
from benchmark_roofline import partition


def phase(step,kind,start,end,rid='r'):
    return dict(event='phase',step=step,phase=kind,start=start,work_start=start+.1,end=end,
                rows=[dict(request=rid,start=0,length=4096,final=True)] if kind=='prefill'
                else [dict(request=rid,position=4096)])


def admit(rid,at):
    return dict(event='admitted',request=rid,start=at-.01,end=at)


def consume(step,start,end,*,rid='r',finish=False):
    return dict(event='consumed',step=step,start=start,end=end,requests=[rid],finished=[rid] if finish else [])


def own(pieces,start,end,kind):
    actual=[x for x in pieces if x['start']<end and x['end']>start]
    assert actual and all(x['phase']==kind for x in actual),(start,end,kind,actual)


def rejects(call,message):
    try: call()
    except ValueError as exc:
        assert message in str(exc),(message,str(exc))
    else: raise AssertionError('Expected rejection: '+message)


# D1 ready5; P2 enters3, drain through5.2, builds through7, ready10.
# D1 consumption at10.1 must not stretch decode across P2 build or execution.
a=[phase(1,'decode',1,5),phase(2,'prefill',3,10)]
e=[admit('r',.5),consume(1,10.1,10.2),consume(2,10.3,10.4,finish=True),
   dict(event='drain',step=2,start=3.2,end=5.2,origin_steps=[1],origin_phases=['decode']),
   dict(event='build_input',step=2,start=5.2,end=7,phase='prefill'),
   dict(event='state_apply',step=1,phase='decode',start=5.5,end=5.6,requests=['r'])]
p=partition(a,e,{'r'})
own(p,3,5.2,'decode');own(p,5.2,5.5,'prefill');own(p,5.5,5.6,'decode')
own(p,5.6,10,'prefill');own(p,10.1,10.2,'decode')

# P1 ready3, D2 scheduled2 and ready7; older P tail consumed later must
# leave D scheduling/preparation after3 assigned to D, except its own tagged tail.
a=[phase(1,'prefill',1,3),phase(2,'decode',2,7)]
e=[admit('r',.5),consume(1,4,4.2),consume(2,7.2,7.3,finish=True),
   dict(event='state_apply',step=1,phase='prefill',start=3.3,end=3.4,requests=['r'])]
p=partition(a,e,{'r'})
own(p,2,3,'prefill');own(p,3,3.3,'decode');own(p,3.3,3.4,'prefill')
own(p,3.4,4,'decode');own(p,4,4.2,'prefill');own(p,4.2,7,'decode')

# Overlapped decode is unioned. Active same-phase gap7→10 remains decode.
a=[phase(1,'decode',1,5),phase(2,'decode',3,7),phase(3,'decode',10,12)]
e=[admit('r',.5),consume(1,5.1,5.2),consume(2,7.1,7.2),consume(3,12.1,12.2,finish=True)]
p=partition(a,e,{'r'});own(p,1,12.2,'decode')
assert abs(sum(x['end']-x['start'] for x in p if x['phase']=='decode')-11.2)<1e-10

# B1 request-free HTTP/client turnaround cannot be charged to the next request.
a=[phase(1,'decode',1,2,'a'),phase(2,'prefill',6,8,'b')]
e=[admit('a',.5),consume(1,2.1,2.2,rid='a',finish=True),admit('b',5),consume(2,8.1,8.2,rid='b',finish=True)]
p=partition(a,e,{'a','b'});own(p,2.2,5,'residual');own(p,5,6,'prefill')
# Unexecuted terminal state-apply is not required or invented; a later span is
# outside completed client scope and omitted, while its marker stays raw.
e.append(dict(event='state_apply',step=2,phase='prefill',start=20,end=21,requests=['b']))
p=partition(a,e,{'a','b'},window=(.4,8.3));own(p,8.2,8.3,'residual')
assert max(x['end'] for x in p)==8.3

# Reject incomplete, mixed, ambiguous and out-of-window evidence.
rejects(lambda:partition(a,e[:-2],{'a','b'}),'Missing admission/finished')
mixed=[phase(1,'decode',1,2,'a')];mixed[0]['rows'].append(dict(request='foreign',position=4096))
rejects(lambda:partition(mixed,e,{'a'}),'Mixed measured/foreign')
intermediate=[phase(1,'prefill',1,2)];intermediate[0]['rows'][0]['final']=False
rejects(lambda:partition(intermediate,e,{'r'}),'Intermediate prefill')
foreign=e+[phase(9,'decode',1.5,1.7,'foreign')]
rejects(lambda:partition(a,foreign,{'a','b'},window=(.4,8.3)),'Foreign execution')
rejects(lambda:partition(a,e,{'a','b'},window=(.4,7.9)),'escapes')
print('PASS: D→P drain/build ownership; P→D delayed consumption; state-apply carving;')
print('same-phase overlap and active gaps; HTTP idle residuals; terminal queued bookkeeping; rejection cases')

# A finished warmup's deferred CPU state update can run inside the first measured
# build. Its explicitly tagged span is residual, despite measured base service.
a=[phase(11,'prefill',1,4)]
e=[admit('r',.5),consume(11,4.1,4.2,finish=True),
   dict(event='state_apply',step=10,phase='decode',enclosing_step=11,
        start=1.5,end=1.7,requests=['warmup']),
   consume(10,2,2.1,rid='warmup',finish=True)]
p=partition(a,e,{'r'})
own(p,1,1.5,'prefill');own(p,1.5,1.7,'residual');own(p,1.7,2,'prefill')
own(p,2,2.1,'residual');own(p,2.1,4,'prefill')
assert any(x['reason']=='foreign_state_apply' for x in p)
assert any(x['reason']=='foreign_consumed' for x in p)
print('PASS: foreign warmup/probe host bookkeeping carved out as explicit residual')
