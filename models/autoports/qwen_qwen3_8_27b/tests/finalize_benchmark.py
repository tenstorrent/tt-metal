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
    identity=json.loads((BASE/'identity.json').read_text()) if (BASE/'identity.json').exists() else {}
    setup=json.loads((BASE/'setup.json').read_text())
    packet=json.loads(PACKET.with_name(PACKET.stem+'.template.json').read_text())
    packet.update(selected_mesh=[1,4],performance={'source':'vllm'},accuracy=[],
        missing_reasons={'device_metrics':'Live vLLM device profiling is prohibited. Host-wall phase rooflines are reported separately and are not device-time rooflines.'},
        notes=['Full64-layer autoport; head_bfp8_hifi2; four Blackhole p300c ASICs.',
               'GPQA common10/198 subset, native thinking xhigh, 16384 total-output cap; not full-dataset accuracy.',
               'SWE-bench Verified and Terminal-Bench2 need goal12 agent harnesses.',
               'Decode DRAM accounting is an explicit estimate; no hardware bus counters.'])
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
    report+='Document IDs, content/population hashes and few-shot settings are frozen in [manifest.json](manifest.json); '
    report+='[protocol.json](../protocol.json) records the exact variant and seeds. The public GPQA mirror matches the packaged198-document population hash exactly. '
    report+='Published89.2% is for the exact model, but the full evaluation recipe and output budget are not proven equivalent to this subset protocol. '
    report+='No accuracy acceptance verdict is assigned.\n\n'
    report+=f"Implementation: `{identity.get('imported_module_file')}`, {identity.get('layer_count')}/{identity.get('configured_layer_count')} layers. "
    report+=f"Hardware: {identity.get('hardware')}. Selected precision: head_bfp8_hifi2, BFP4 decoder weights, BFP8 head/KV, LoFi projections, HiFi2 head, FP32 recurrence. "
    report+='Native context remains262144 on every profile; prefix caching is enabled. Distinct prompts and exact server-reported4096/128 token counts are required by the runner. '
    report+='The HTTP path includes an owned loopback TCP relay to the existing Docker bridge.\n\n'
    report+='Source commits: `'+json.dumps(identity.get('source_commits',{}),sort_keys=True)+'`. '
    report+='The installed vLLM0.30.0+empty wheel does not expose its source commit; [installed-source fingerprint](vllm-installed-source.json) records its Python source-tree hash. '
    report+='The benchmark client is upstream vLLM0.26.0+empty in a separate environment; accuracy uses lm-eval0.4.13. '
    report+='Each profile’s configuration snapshot retains observed process argv, effective server settings, imported autoport marker, full layer list, precision, tokenizer hashes and hardware initialization.\n\n'
    report+='Roofline denominators are disjoint full host phase intervals through existing completion and scheduler consumption, including host work, communication and gaps. '
    report+='Unique client IDs exclude initial probes and warmups. No device synchronization or live profiler was added. '
    report+='Raw events and request/step mappings are retained per profile. Useful FLOPs omit padding; estimated DRAM bytes include shared weight reads, KV/state and explicit material intermediates, but omit unknown rereads/spills. '
    report+='The theoretical peak uses four120-core ASICs at nominal1.35GHz and four512GB/s DRAM interfaces; observed startup clock settled at1331MHz. '
    report+='These are modeled host-wall utilization estimates, not device-time utilization or measured bus bandwidth.\n\n'
    report+='No ready server was supplied: Stage10’s cleanup had stopped it. The first launch and all subsequent launches/warmups were inside the client timer. '
    report+='Pre-invocation setup (client installation, downloads, subset preparation and collector implementation) is recorded separately in [setup.json](../setup.json); '
    report+='its complete elapsed time and the prior Stage10 startup duration were not measured. '
    report+='The four configured batches allow7200 seconds total. See [RUN_NOTES.md](../RUN_NOTES.md) for repairs, checks and local commit receipts.\n'
    (RUN/'REPORT.md').write_text(report)
    print(json.dumps(dict(status=summary['status'],packet=str(PACKET),accuracy_records=len(packet['accuracy']))))


if __name__=='__main__':
    main()
