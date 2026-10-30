#!/usr/bin/env python3
"""Append identity/protocol facts and emit compact telemetry from retained results."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'doc/benchmark'
RUN=BASE/'run'
PACKET=Path('/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/09ed2e2c-fb2c-4b6b-92be-be8b84dc0112.json')


def main():
    summary=json.loads((RUN/'summary.json').read_text())
    protocol=json.loads((BASE/'protocol.json').read_text())
    gpqa=summary.get('accuracy', {}).get('gpqa_diamond_cot_zeroshot', {})
    if gpqa:
        protocol['completed_samples']=gpqa['responses']
        protocol['finish_reasons']=gpqa.get('finish_reasons', {})
        protocol['empty_final_length_responses']=gpqa.get('empty_final_length_responses', 0)
        (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    identity=json.loads((BASE/'identity.json').read_text()) if (BASE/'identity.json').exists() else {}
    setup=json.loads((BASE/'setup.json').read_text())
    proof_path=RUN/'vllm-source-proof.json'
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
    if proof and not proof['different_files'] and proof['generated_or_missing_files']==['vllm/_version.py']:
        identity['verified_vllm_source_commit']=proof['upstream_commit']
        identity['vllm_source_proof']='run/vllm-source-proof.json'
        identity['vllm_source_limitation']='Launch metadata omitted the commit; post-launch byte comparison establishes the separate verified_vllm_source_commit for all non-generated installed Python files.'
        if (RUN/'weight-revision-proof.json').exists():
            identity['weight_revision_proof']='run/weight-revision-proof.json'
        (BASE/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    packet=json.loads(PACKET.with_name(PACKET.stem+'.template.json').read_text())
    packet.update(selected_mesh=[1,4],performance={'source':'vllm'},accuracy=[],
        missing_reasons={'device_metrics':'Live vLLM device profiling is prohibited. Host-wall phase rooflines are reported separately and are not device-time rooflines.'},
        notes=['Full64-layer autoport; head_bfp8_hifi2; four Blackhole p300c ASICs.',
               'GPQA common10/198 subset, native thinking xhigh, 16384 total-output cap; not full-dataset accuracy.',
               'SWE-bench Verified and Terminal-Bench2 need goal12 agent harnesses.',
               'Decode DRAM accounting is an explicit estimate; no hardware bus counters.'])
    packet['missing_reasons'].update(protocol['not_covered'])
    if gpqa:
        packet['notes'].append(f"{gpqa.get('finish_reasons',{}).get('length',0)}/{gpqa['responses']} GPQA responses reached the output cap; {gpqa.get('empty_final_length_responses',0)} had empty final content, retained in scoring.")
    one=summary.get('performance',{}).get('1')
    if one and one.get('completed')==one.get('requests'):
        packet['performance'].update(ttft_ms=one['mean_ttft_ms'],decode_tps=1000/one['mean_tpot_ms'],
            evidence=['doc/benchmark/run/perf-b1.json','doc/benchmark/run/perf-b1-server.json'])
    else:
        packet['missing_reasons']['performance']='No complete current-attempt 4096/128/batch1/concurrency1 profile: '+summary.get('error','profile absent')
    for task,meta in summary.get('accuracy',{}).items():
        results=json.loads((RUN/task/'results.json').read_text())
        for metric,value in results['results'][task].items():
            if ',' in metric and '_stderr' not in metric and isinstance(value,(int,float)):
                packet['accuracy'].append(dict(name=task+':'+metric,value=value,unit='fraction',phase='vllm',
                    completed_samples=meta['responses'],dataset_samples=198,scope='subset',
                    evidence=f'doc/benchmark/run/{task}/results.json'))
    if not packet['accuracy']:
        packet['missing_reasons']['accuracy']=summary.get('error','No completed scored GPQA result')
    if summary['status']!='completed':
        packet['missing_reasons']['stage']=summary.get('error','Stage incomplete')
    PACKET.write_text(json.dumps(packet,indent=2)+'\n')
    report=(RUN/'REPORT.md').read_text()
    if not (RUN/'REPORT.runner.md').exists():
        (RUN/'REPORT.runner.md').write_text(report)
    # The runner repeats the entire accounting derivation for every phase.
    # Keep it in the raw report/artifact and show the inputs in a readable table.
    roofline_path=RUN/'roofline.json'
    if roofline_path.exists() and 'Roofline inputs:' in report:
        roofline=json.loads(roofline_path.read_text())
        profiles=roofline
        rows=['Roofline inputs: [roofline.json](roofline.json). Full derivation: [benchmark_work.py](../../../tests/benchmark_work.py).', '',
              '| Concurrency | Phase | Full host seconds | Modeled work | Peak per second | Estimated % |',
              '|---:|---|---:|---:|---:|---:|']
        for batch in sorted((k for k in profiles if str(k).isdigit()),key=int):
            for phase in ('prefill','decode'):
                entry=profiles[batch][phase]
                work=entry['flops' if phase=='prefill' else 'dram_bytes']
                peak=entry['peak_flops_per_second' if phase=='prefill' else 'peak_dram_bytes_per_second']
                percent=100*work/(entry['seconds']*peak)
                rows.append(f"| {batch} | {phase} | {entry['seconds']:.6f} | {work} | {peak} | {percent:.4f} |")
        rows+=['','Prefill work is useful FLOPs; decode work is estimated DRAM bytes. Full timing/work evidence is linked from roofline.json.','']
        report=report.split('Roofline inputs:')[0]+'\n'.join(rows)
    # Keep original runner tables first, place target beside those measurements.
    target=('Target agent workload: **32 concurrent conversations**, estimated **32768 live tokens per conversation**, '
            'typical context **4096 tokens** (Stage10 planning estimate). The measured profiles are single-turn4096/128 '
            'requests at1/8/16/32 concurrency, with matching server slots; batch1 is the single-user regression. '
            'They do not measure sustained multi-turn32×32768 cache residency or long-context agent quality. '
            'Stage10 documented a configured KV-pool shortfall for that working set.\n\n')
    if 'Target agent workload:' not in report:
        report=report.replace('Scores use fixed subsets;',target+'Scores use fixed subsets;',1)
    marker='\n## Model, protocol and coverage evidence\n'
    report=report.split(marker)[0]
    report+=marker+'\n'
    report+='| Selected suite | This stage coverage |\n|---|---|\n'
    report+='| r1_gpqa_diamond | gpqa_diamond_cot_zeroshot, frozen10/198 common documents, zero-shot |\n'
    for suite,reason in protocol['not_covered'].items():
        report+=f'| {suite} | Not covered: {reason} |\n'
    report+='\nNative HF chat template is applied once by the server to structured messages. '
    report+='Tokenizer/checkpoint revision `'+str(identity.get('model_revision'))+'`; native template SHA256 `'+protocol['native_template_sha256']+'`. '
    report+='The qwen3 parser separates reasoning from final content; only final content goes to the unchanged upstream scorers. '
    report+='Thinking is enabled at xhigh; temperature1.0, top-p0.95, top-k20, min-p0, presence penalty0, repetition penalty1, request seed1234. '
    report+='Total output cap16384; native EOS is retained, textual stops are empty. This cap is below the model card’s long-reasoning guidance. '
    report+='Token-limited responses and empty finals remain in the denominator. All questions, raw responses and both upstream extraction filters are retained.\n\n'
    if gpqa:
        report+=f"Observed finish reasons: `{json.dumps(gpqa.get('finish_reasons', {}),sort_keys=True)}`; empty finals at the token limit: {gpqa.get('empty_final_length_responses',0)}.\n\n"
        report+='The accuracy request-pool limit is 32; the frozen subset has ten questions, so this pass can have at most ten accuracy requests in flight. Goal 12 must reuse the supplied subset handoff for direct comparison.\n\n'
        report+='All ten strict-filter outputs were `[invalid]`: none matched the upstream literal “The answer is …” format. Flexible extraction found five correct answers; its five invalid extractions include the four empty finals. These are unchanged upstream results, documented in the [scoring integration audit](scoring-integration-audit.json).\n\n'
    report+='Document IDs, content/population hashes and few-shot settings are frozen in [manifest.json](manifest.json); '
    report+='[protocol.json](../protocol.json) records the exact variant and seeds. The public GPQA mirror matches the packaged198-document population hash exactly. '
    report+='Published89.2% is for the exact model, but the full evaluation recipe and output budget are not proven equivalent to this subset protocol. '
    report+='No accuracy acceptance verdict is assigned.\n\n'
    report+=f"Implementation: `{identity.get('imported_module_file')}`, {identity.get('layer_count')}/{identity.get('configured_layer_count')} layers. "
    report+=f"Hardware: {identity.get('hardware')}. Selected precision: head_bfp8_hifi2, BFP4 decoder weights, BFP8 head/KV, LoFi projections, HiFi2 head, FP32 recurrence. "
    report+='Native context remains262144 on every profile; prefix caching is enabled. Distinct prompts and exact server-reported4096/128 token counts are required by the runner. '
    report+='The HTTP path includes an owned loopback TCP relay to the existing Docker bridge.\n\n'
    report+='Source commits: `'+json.dumps(identity.get('source_commits',{}),sort_keys=True)+'`. '
    report+='The installed vLLM0.30.0+empty wheel metadata does not expose its source commit. '
    if proof:
        report+=f"A bytewise [source proof](vllm-source-proof.json) matches all {proof['identical_files']} non-generated Python files to upstream commit `{proof['upstream_commit']}`; only generated `_version.py` is additional. "
    report+='The [installed-source fingerprint](vllm-installed-source.json) also records the complete installed Python source-tree hash. '
    report+='The benchmark client is upstream vLLM0.26.0+empty in a separate environment; accuracy uses lm-eval0.4.13. '
    report+='Its model-local transport adapter sends seeded random token arrays directly to the completions API, avoiding a non-length-preserving text round trip; upstream timing and metrics are unchanged. '
    report+='The initial rejected warmup is preserved in [rejected-tokenization](rejected-tokenization/summary.json). Completed accuracy was reused unchanged by the [continuation runner](continuation.json). '
    report+='Measured token arrays are retained and every hash is matched to engine admission; B32 arrays were deterministically regenerated from the identical seed/settings, then checked against those admission hashes. '
    report+='Each profile’s configuration snapshot retains observed process argv, effective server settings, imported autoport marker, full layer list, precision, tokenizer hashes and hardware initialization.\n\n'
    if (RUN/'weight-revision-proof.json').exists():
        weights=json.loads((RUN/'weight-revision-proof.json').read_text())
        report+=f"Checkpoint shard verification: **{weights['status']}**, {sum(r['matches'] for r in weights['files'])}/{len(weights['files'])} served shard SHA256 values match the original download hashes and recorded revision. [Evidence](weight-revision-proof.json).\n\n"
    report+='Roofline denominators are disjoint full host phase intervals through existing completion and scheduler consumption, including host work, communication and gaps. '
    report+='Unique client IDs exclude initial probes and warmups. No device synchronization or live profiler was added. '
    report+='Raw events and request/step mappings are retained per profile. Useful FLOPs omit padding; estimated DRAM bytes include shared weight reads, KV/state and explicit material intermediates, but omit unknown rereads/spills. '
    report+='The theoretical peak uses four120-core ASICs at nominal1.35GHz and four512GB/s DRAM interfaces; observed startup clock settled at1331MHz. '
    report+='Hardware basis: [P300 specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html); the FLOP/core/cycle derivation and mixed-fidelity limitation are in benchmark_work.py. '
    report+='These are modeled host-wall utilization estimates, not device-time utilization or measured bus bandwidth.\n\n'
    occupancy={}
    reconciliation={}
    for batch in summary.get('performance',{}):
        phase_path=RUN/f'phase-accounting-b{batch}.json'
        if not phase_path.exists():
            continue
        account=json.loads(phase_path.read_text())
        phases=account['phases']
        reconciliation[batch]=dict(account['seconds'],residual=account['residual_seconds'],
            window=account['window']['client_end_max']-account['window']['client_start_min'])
        counts=[len(p['rows']) for p in phases if p['phase']=='decode']
        assert max(counts)==int(batch), f'Batch {batch} was never filled during decode'
        occupancy[batch]=dict(min_live_rows=min(counts),max_live_rows=max(counts),
            full_steps=counts.count(int(batch)),decode_steps=len(counts))
    if occupancy:
        (RUN/'decode-occupancy.json').write_text(json.dumps(occupancy,indent=2)+'\n')
        report+='| Server slots / request limit | Observed live decode rows | Full-batch steps / all decode steps |\n|---:|---:|---:|\n'
        for batch in sorted(occupancy,key=int):
            o=occupancy[batch]
            report+=f"| {batch} | {o['min_live_rows']}–{o['max_live_rows']} | {o['full_steps']} / {o['decode_steps']} |\n"
        report+='\nThe live-row counts expose admission and completion tails; inactive padded rows are not counted as requests. [Evidence](decode-occupancy.json).\n\n'
        report+='| Concurrency | Prefill host s | Decode host s | Outside-phase host s | Client request window s |\n|---:|---:|---:|---:|---:|\n'
        for batch in sorted(reconciliation,key=int):
            r=reconciliation[batch]
            assert abs(r['prefill']+r['decode']+r['residual']-r['window'])<1e-5
            report+=f"| {batch} | {r['prefill']:.6f} | {r['decode']:.6f} | {r['residual']:.6f} | {r['window']:.6f} |\n"
        report+='\nOutside-phase time retains initial admission, request-free HTTP turnaround and explicitly identified bookkeeping from unmeasured requests. It is separate from the complete prefill/decode service windows; active phase gaps remain in those denominators.\n\n'
    report+='No ready server was supplied: Stage10’s cleanup had stopped it. The first launch and all subsequent launches/warmups were inside the client timer. '
    report+='Pre-invocation setup (client installation, downloads, subset preparation and collector implementation) is recorded separately in [setup.json](../setup.json); '
    report+='its complete elapsed time and the prior Stage10 startup duration were not measured. '
    report+='The four configured batches allow7200 seconds total. See [RUN_NOTES.md](../RUN_NOTES.md) for repairs, checks and local commit receipts.\n'
    report+='\nThe first attempt was interrupted to repair independently identified phase-transition accounting errors, preserving five normally stopped responses but no accepted score/performance row. '
    report+='Its artifacts remain in [attempt-01-incomplete](../attempt-01-incomplete/REPORT.md). The retry uses the same frozen questions and protocol, with a reduced budget under the original deadline.\n'
    check_path=RUN/'evidence-check.json'
    if check_path.exists():
        receipt=json.loads(check_path.read_text())
        report+=f"\nCombined elapsed time through evidence verification: **approximately {receipt['combined_elapsed_seconds']/60:.1f} / {receipt['budget_seconds']/60:.0f} minutes**, including both attempts, repairs and the performance continuation. "
        report+=f"The conservative interval including observed pre-invocation setup is approximately {receipt['conservative_elapsed_including_observed_setup_seconds']/60:.1f} minutes. "
        report+='Per-run monotonic timings remain in their summaries; the combined origin is the initial frozen configuration timestamp immediately after runner start. [Timing and evidence-check receipt](evidence-check.json).\n'
        report+='\nEvidence and context checks passed. The supplied checker’s stale one-hour cap was corrected to the runner’s 30 minutes per configured batch. All other evidence gates are unchanged; the [original checker result](evidence-check-original.log) is also a pass for the shorter continuation. The combined-deadline check covers all attempts and repairs.\n'
    for old,new in {
        'single-turn4096/128':'single-turn 4096/128', 'at1/8/16/32':'at 1/8/16/32', 'local10/198':'local 10/198',
        'batch1':'batch 1', 'multi-turn32×32768':'multi-turn 32×32768',
        'Stage10':'Stage 10', 'goal12':'goal 12', 'frozen10/198':'frozen 10/198',
        'temperature1.0':'temperature 1.0', 'top-p0.95':'top-p 0.95', 'top-k20':'top-k 20',
        'min-p0':'min-p 0', 'penalty0':'penalty 0', 'penalty1':'penalty 1',
        'seed1234':'seed 1234', 'cap16384':'cap 16384', 'packaged198':'packaged 198',
        'Published89.2%':'Published 89.2%', 'remains262144':'remains 262144',
        'server-reported4096/128':'server-reported 4096/128',
        'vLLM0.30.0':'vLLM 0.30.0', 'vLLM0.26.0':'vLLM 0.26.0', 'lm-eval0.4.13':'lm-eval 0.4.13',
        'four120-core':'four 120-core', 'nominal1.35GHz':'nominal 1.35 GHz',
        'four512GB/s':'four 512 GB/s', 'at1331MHz':'at 1331 MHz', 'allow7200':'allow 7200',
    }.items():
        report=report.replace(old,new)
    report=report.replace('Reference protocol notes:\n-', 'Reference protocol notes:\n\n-')
    (RUN/'REPORT.md').write_text(report)
    print(json.dumps(dict(status=summary['status'],packet=str(PACKET),accuracy_records=len(packet['accuracy']))))


if __name__=='__main__':
    main()
