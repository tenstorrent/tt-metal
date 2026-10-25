# SPDX-License-Identifier: Apache-2.0
"""Verify complete acceptance and source provenance on the selected default."""
import hashlib
import json
import math
import re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
b=root/'doc/optimized_multichip_decoder'
sha=hashlib.sha256((root/'tt/multichip_decoder.py').read_bytes()).hexdigest()
cases={f'final_accepted_l{l}':False for l in (0,3)}
cases['packed_prefill_multichunk_b8_l3']=False
for suite in ('batch','boundary','continuation','context','stack','watcher'):
    rows=json.loads((b/f'{suite}_validation.json').read_text())
    expected_count={'batch':8,'boundary':38,'continuation':4,'context':2,'stack':6,'watcher':4}[suite]
    assert len(rows)==expected_count and all(row['exit_code']==0 for row in rows),suite
    for row in rows:
        cases[row['case']]=False
for l in (0,3):cases[f'final_reserved_context_l{l}']=False
checks=[]
for case,unchanged in cases.items():
    r=json.loads((b/f'{case}.json').read_text());m=r['multichip']
    expected=sha
    assert r['provenance']['tt/multichip_decoder.py']==expected,case
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in r['provenance'].items()),case
    scores=[r['prefill_pcc'],*r['decode_pcc'],*r.get('state_pcc',{}).values()]
    assert all(math.isfinite(x) and x>=.995 for x in scores),case
    assert m['runtime_fallback_audit'] and m['restored_replays_bitwise']==3,case
    assert m['unique_ccl_contexts']==1,case
    if r['batch']>=8:assert m['decode_residual_layout']=='sharded',case
    if r['stack']:
        assert m['scratch_workspace']['unique_pools']==1 and m['scratch_workspace']['owners']==2,case
        assert m['async_stress']['blocking_vs_nonblocking_bitwise'],case
        assert m['async_stress']['program_cache_allocations_included'],case
    if case.startswith('watcher'):
        assert m['watcher_stress_replays']>0 and (b/f'{case}.watcher.txt').exists(),case
        watcher=(b/f'{case}.watcher.txt').read_text()
        assert re.search(r'Dump #\d+ completed',watcher),case
        assert all(f'detach device {device}' in watcher for device in range(4)),case
        assert not re.search(r'assertion failed|corruption|invalid noc|watcher error|watcher detected',watcher,re.I),case
    checks.append(dict(case=case,layer_type=r['layer_type'],batch=r['batch'],seq=r['seq'],steps=r['steps'],
        prefill_pcc=r['prefill_pcc'],minimum_decode_pcc=min(r['decode_pcc']),minimum_state_pcc=min(r['state_pcc'].values()) if r.get('state_pcc') else None,
        source='final source',
        runtime_fallback_audit=True,evidence=f'{case}.json'))
traces=[]
for name in ('final_workspace_lifetime','mixed_traces_l0','mixed_traces_l3','mixed_prefill_decode_traces'):
    r=json.loads((b/f'{name}.json').read_text())
    assert r['exact'] and r['allocation_tracking'] and r['program_cache_tracking'] and not r['acknowledgments'],name
    assert r['nonblocking_replays']>=259,name
    if name=='final_workspace_lifetime':
        assert r['workspace_sha256']==hashlib.sha256((root/'tt/decode_workspace.py').read_bytes()).hexdigest()
    if name=='mixed_prefill_decode_traces':
        assert r['shared_owners']==1 and r['stable_persistent_addresses'] and r['captures_before_traffic']==11
        assert r['prefill_rs_bytes_per_device']==55705600
        assert all(hashlib.sha256((root/path).read_bytes()).hexdigest()==digest for path,digest in r['provenance'].items())
    elif name.startswith('mixed'):
        assert r['provenance']['tt/multichip_decoder.py']==sha
        assert r['decode_layouts']==['replicated','replicated','sharded','sharded','sharded']
    traces.append(dict(evidence=f'{name}.json',captures=r['captures_before_traffic'],nonblocking_replays=r['nonblocking_replays'],exact=True,allocation_tracking=True,program_cache_tracking=True))
r=json.loads((b/'packed_conv_exact.json').read_text())
assert r['exact'] and r['decoder_sha256']==sha and len(r['checks'])==18
for name,count in [('final_packed_rope_exact',54),('final_prefill_rope_exact',14)]:
    r=json.loads((b/f'{name}.json').read_text())
    assert r['exact'] and len(r['checks'])==count,name
    assert all(hashlib.sha256((root/path).read_bytes()).hexdigest()==digest for path,digest in r['provenance'].items()),name
profiles=[]
for label in ('final_profile','final_b32_profile'):
    for l in (0,3):
        name=f'{label}_l{l}';r=json.loads((b/f'{name}.json').read_text())
        assert r['provenance']['tt/multichip_decoder.py']==sha,name
        assert all(hashlib.sha256((root/path).read_bytes()).hexdigest()==digest for path,digest in r['provenance'].items()),name
        timing=json.loads((b/f'tracy/{name}/whole_layer_timing.json').read_text())
        assert all(set(v['per_device'])=={'0','1','2','3'} for v in timing['phases'].values()),name
        profiles.append(dict(evidence=f'tracy/{name}/whole_layer_timing.json',source_matches=True,all_four_devices=True))
review=b/'stage_review.md'
clean=review.exists() and 'Verdict: clean-pass' in review.read_text()
result=dict(status='all_acceptance_checks_passed_clean_review' if clean else 'all_acceptance_checks_passed_review_pending',decoder_sha256=sha,
    pcc_gate=.995,measured_path='TP4 mesh1x4; partitioned weights and state, no single-chip fallback',
    cases=checks,trace_lifetime_checks=traces,exact_convolution_check='packed_conv_exact.json',exact_rope_checks=['final_packed_rope_exact.json','final_prefill_rope_exact.json'],profiles=profiles,
    inter_layer_contract='residual_contract.md; DecoderStack calls the next decoder directly',
    scope='Decoder layers and two-layer mixed-kind stacks; no full-model or serving claim')
(b/'validation_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(len(checks),'acceptance cases;',len(profiles),'profiles;',sum(x['nonblocking_replays'] for x in traces),'nonblocking workspace/mixed replays')
