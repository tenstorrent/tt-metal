# SPDX-License-Identifier: Apache-2.0
"""Opt-in host-only phase evidence; no device operations or per-step file writes.

Installed on the actual plugin runner after model construction. Existing async
finalization supplies completion, never trace enqueue. Buffered records are
flushed by SIGUSR1 only when the collector has finished its HTTP workload.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def install(model):
    path = os.environ.get('QWEN38_BENCHMARK_PHASES')
    if not path:
        return
    from vllm_tt_plugin.model_runner import TTModelRunner
    from vllm_tt_plugin.async_decode import TTAsyncDecodeController, AsyncTTModelRunnerOutput
    from vllm_tt_plugin.scheduler import TTScheduler
    if getattr(TTModelRunner, '_qwen_benchmark_installed', False):
        raise RuntimeError('Duplicate benchmark phase installation')
    TTModelRunner._qwen_benchmark_installed = True
    events, requests, pending = [], {}, {}
    serial = [0]
    current = [None]
    scheduled = {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def flush(*_):
        # Runs on the main Python thread; collector signals only an idle server.
        snapshot = list(events)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(e)+'\n' for e in snapshot))
        temporary.replace(path)
        path.with_suffix('.flush.json').write_text(json.dumps(dict(pending_records=len(pending),
            completed_records=len(snapshot),flushed=time.perf_counter()))+'\n')

    if signal.getsignal(signal.SIGUSR1) not in (signal.SIG_DFL, signal.SIG_IGN):
        raise RuntimeError('SIGUSR1 already has an owner')
    signal.signal(signal.SIGUSR1, flush)
    ready = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                 module=__file__, module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 batch=model.batch_size, layers=model.model.layer_indices,
                 precision=model.model.precision_config, mesh=[1,4],
                 hooks=['execute_model','_forward_with_model_input','_finish_front_packed_sync_impl',
                        'submit_async_decode','complete_decode_step',
                        'AsyncTTModelRunnerOutput._get_output_impl'])
    path.with_suffix('.ready.json').write_text(json.dumps(ready,indent=2)+'\n')

    def wrap(cls, name, wrapper):
        original = getattr(cls, name)
        setattr(cls, name, functools.wraps(original)(wrapper(original)))

    def execute(original):
        def call(runner, scheduler_output):
            serial[0] += 1
            start=scheduled.get(id(scheduler_output),time.perf_counter())
            scheduled[id(scheduler_output)]=(start,serial[0])
            current[0] = dict(step=serial[0], start=start,
                              scheduled=dict(scheduler_output.num_scheduled_tokens))
            return original(runner, scheduler_output)
        return call

    def schedule(original):
        def call(scheduler, *args, **kwargs):
            start=time.perf_counter()
            result=original(scheduler,*args,**kwargs)
            if result.num_scheduled_tokens:
                scheduled[id(result)]=start
            return result
        return call

    def consume(original):
        def call(scheduler, scheduler_output, model_runner_output):
            result=original(scheduler,scheduler_output,model_runner_output)
            timing=scheduled.pop(id(scheduler_output),None)
            if isinstance(timing,tuple):
                events.append(dict(event='consumed',step=timing[1],end=time.perf_counter(),
                    requests=list(scheduler_output.num_scheduled_tokens)))
            return result
        return call

    def snapshot(runner, inputs):
        record = dict(current[0])
        record['work_start'] = time.perf_counter()
        record['phase'] = 'decode' if inputs.prompt_lens is None else 'prefill'
        ids = list(inputs.row_req_ids or runner.input_batch.req_ids[:runner.input_batch.num_reqs])
        positions = inputs.input_positions.tolist()
        rows = []
        for i, rid in enumerate(ids):
            state = runner.requests[rid]
            if rid not in requests:
                tokens = list(state.prompt_token_ids)
                requests[rid] = dict(request=rid, first_seen=record['start'],
                    prompt_tokens=len(tokens), prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                events.append(dict(event='request', **requests[rid]))
            if inputs.prompt_lens is not None:
                start, end = int(positions[i]), int(inputs.prompt_lens[i])
                final = not bool(inputs.intermediate_prefill_mask[i])
                rows.append(dict(request=rid,start=start,length=end-start,final=final))
                requests[rid]['next_decode_position'] = end
            else:
                # Resident token feedback can be one host step stale. The
                # request's executed sequence is advanced at submission.
                pos = requests[rid]['next_decode_position']
                rows.append(dict(request=rid,position=pos,host_position=int(positions[i])))
                requests[rid]['next_decode_position'] = pos+1
        record.update(rows=rows,batch=model.batch_size,mesh=[1,4])
        return record

    def sync_forward(original):
        def call(runner, inputs):
            record = snapshot(runner, inputs)
            result = original(runner, inputs)
            pending[id(result)] = record
            return result
        return call

    def sync_finish(original):
        def call(runner, grammar_output, *, fwd):
            record = pending.pop(id(fwd), None)
            result = original(runner, grammar_output, fwd=fwd)
            if record is not None:
                record.update(event='phase',end=time.perf_counter(),completion='sync_sample_and_runner_output')
                events.append(record)
            return result
        return call

    def submit(original):
        def call(controller, inputs, **kwargs):
            record = snapshot(controller.runner, inputs)
            result = original(controller, inputs, **kwargs)
            # The context is immutable and is retained by the async wrapper.
            pending[id(result._context)] = record
            return result
        return call

    def complete(original):
        def call(controller, submission, model_input, context, **kwargs):
            result = original(controller, submission, model_input, context, **kwargs)
            # Keep the controller's actual readback boundary, but do not close
            # the whole phase until output construction and enqueue also finish.
            record = pending[id(context)]
            record.update(submit_ns=context.submit_time_ns,complete_ns=result.completion_time_ns)
            return result
        return call

    def output_complete(original):
        def call(output):
            result = original(output)
            record = pending.pop(id(output._context))
            record.update(event='phase',end=time.perf_counter(),
                          completion='async_readback_sampling_and_runner_output_complete')
            events.append(record)
            return result
        return call

    wrap(TTModelRunner,'execute_model',execute)
    wrap(TTScheduler,'schedule',schedule)
    wrap(TTScheduler,'update_from_output',consume)
    wrap(TTModelRunner,'_forward_with_model_input',sync_forward)
    wrap(TTModelRunner,'_finish_front_packed_sync_impl',sync_finish)
    wrap(TTAsyncDecodeController,'submit_async_decode',submit)
    wrap(TTAsyncDecodeController,'complete_decode_step',complete)
    wrap(AsyncTTModelRunnerOutput,'_get_output_impl',output_complete)
