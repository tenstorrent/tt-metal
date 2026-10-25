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
import threading
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
    schedule_records = {}
    context_owners = {}
    local = threading.local()
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
                        'AsyncTTModelRunnerOutput._get_output_impl','schedule','add_request',
                        'update_from_output','build_model_input','wait_for_all_pending_async_steps',
                        'enqueue_completed_decode_step','apply_completed_decode_step'])
    path.with_suffix('.ready.json').write_text(json.dumps(ready,indent=2)+'\n')

    def wrap(cls, name, wrapper):
        original = getattr(cls, name)
        setattr(cls, name, functools.wraps(original)(wrapper(original)))

    def execute(original):
        def call(runner, scheduler_output):
            serial[0] += 1
            start=scheduled.get(id(scheduler_output),time.perf_counter())
            scheduled[id(scheduler_output)]=(start,serial[0])
            if id(scheduler_output) in schedule_records:
                schedule_records[id(scheduler_output)]['step']=serial[0]
            current[0] = dict(step=serial[0], start=start,
                              scheduled=dict(scheduler_output.num_scheduled_tokens))
            return original(runner, scheduler_output)
        return call

    def schedule(original):
        def call(scheduler, *args, **kwargs):
            start=time.perf_counter()
            before = scheduler_state(scheduler)
            result=original(scheduler,*args,**kwargs)
            end=time.perf_counter()
            event=dict(event='schedule',start=start,end=end,scheduled=dict(result.num_scheduled_tokens),
                       before=before,after=scheduler_state(scheduler),
                       finished=list(getattr(result,'finished_req_ids',())))
            events.append(event)
            if result.num_scheduled_tokens:
                scheduled[id(result)]=start
                schedule_records[id(result)]=event
            return result
        return call

    def scheduler_state(scheduler):
        return {kind:[r.request_id for r in getattr(scheduler,kind,())]
                for kind in ('running','waiting')}

    def admission(original):
        def call(scheduler, request):
            start=time.perf_counter()
            result=original(scheduler,request)
            events.append(dict(event='admitted',request=request.request_id,start=start,
                end=time.perf_counter(),after=scheduler_state(scheduler)))
            return result
        return call

    def consume(original):
        def call(scheduler, scheduler_output, model_runner_output):
            start=time.perf_counter()
            result=original(scheduler,scheduler_output,model_runner_output)
            end=time.perf_counter()
            finished=set()
            published={}
            for batch in result.values():
                finished.update(getattr(batch,'finished_requests',None) or ())
                for output in batch.outputs:
                    published[output.request_id]=len(output.new_token_ids)
                    if output.finish_reason is not None:
                        finished.add(output.request_id)
            timing=scheduled.pop(id(scheduler_output),None)
            schedule_records.pop(id(scheduler_output),None)
            if isinstance(timing,tuple):
                events.append(dict(event='consumed',step=timing[1],start=start,end=end,
                    requests=list(scheduler_output.num_scheduled_tokens),after=scheduler_state(scheduler),
                    finished=sorted(finished),published_tokens=published))
            return result
        return call

    def build(original):
        def call(runner, scheduler_output, grammar_output):
            start=time.perf_counter()
            result=original(runner,scheduler_output,grammar_output)
            events.append(dict(event='build_input',step=current[0]['step'],start=start,
                end=time.perf_counter(),phase=None if result is None else
                ('decode' if result.prompt_lens is None else 'prefill')))
            return result
        return call

    def drain(original):
        def call(controller):
            start=time.perf_counter()
            owners={r['step']:r['phase'] for r in tuple(pending.values())}
            result=original(controller)
            events.append(dict(event='drain',step=None if current[0] is None else current[0]['step'],
                start=start,end=time.perf_counter(),origin_steps=list(owners),
                origin_phases=list(owners.values())))
            return result
        return call

    def enqueue(original):
        def call(controller, completed):
            key=id(completed.context)
            owner=context_owners.get(key) or getattr(local,'sync_owner',None)
            if owner is not None:
                context_owners[key]=dict(step=owner['step'],phase=owner['phase'])
            events.append(dict(event='state_enqueued',step=None if owner is None else owner['step'],
                phase=None if owner is None else owner['phase'],time=time.perf_counter(),
                requests=list(completed.context.req_ids)))
            return original(controller,completed)
        return call

    def apply_state(original):
        def call(controller, completed, **kwargs):
            start=time.perf_counter()
            owner=context_owners.get(id(completed.context))
            result=original(controller,completed,**kwargs)
            events.append(dict(event='state_apply',step=None if owner is None else owner['step'],
                phase=None if owner is None else owner['phase'],
                enclosing_step=None if current[0] is None else current[0]['step'],
                start=start,end=time.perf_counter(),requests=list(completed.context.req_ids),
                suppressed=list(kwargs.get('suppress_output_req_ids') or ()),
                skipped_state=list(kwargs.get('skip_state_req_ids') or ())))
            context_owners.pop(id(completed.context),None)
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
            previous=getattr(local,'sync_owner',None)
            local.sync_owner=record
            try:
                result = original(runner, grammar_output, fwd=fwd)
            finally:
                local.sync_owner=previous
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
            context_owners[id(result._context)]=dict(step=record['step'],phase=record['phase'])
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
    wrap(TTScheduler,'add_request',admission)
    wrap(TTScheduler,'update_from_output',consume)
    wrap(TTModelRunner,'build_model_input',build)
    wrap(TTModelRunner,'_forward_with_model_input',sync_forward)
    wrap(TTModelRunner,'_finish_front_packed_sync_impl',sync_finish)
    wrap(TTAsyncDecodeController,'submit_async_decode',submit)
    wrap(TTAsyncDecodeController,'complete_decode_step',complete)
    wrap(TTAsyncDecodeController,'wait_for_all_pending_async_steps',drain)
    wrap(TTAsyncDecodeController,'enqueue_completed_decode_step',enqueue)
    wrap(TTAsyncDecodeController,'apply_completed_decode_step',apply_state)
    wrap(AsyncTTModelRunnerOutput,'_get_output_impl',output_complete)
