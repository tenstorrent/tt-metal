# SPDX-License-Identifier: Apache-2.0
"""TT plugin boundary for the Qwen3.8 autoport's canonical generator."""
import os
import secrets
import json
import time
from pathlib import Path

import torch
import ttnn

from .generator import Generator
from .model import Model
from .optimized_decoder import PagedState


class Qwen38ForCausalLM:
    is_hybrid = True
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True
    decode_input_update_contract = 1
    # Device contract probes cover resident inputs, deferred reads and state DMA.
    # Server-level acceptance remains recorded separately under readiness_vllm.
    model_capabilities = {
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "max_device_top_k": 32,
        "supports_async_decode": True,
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
    }

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, tt_data_parallel=1, optimizations=None):
        if tt_data_parallel != 1:
            raise ValueError("This adapter uses all four devices for tensor parallelism")
        ttnn.CONFIG.throw_exception_on_fallback = True
        checkpoint = Path(os.environ.get("QWEN38_CHECKPOINT", ".cache/qwen38-full"))
        indices = os.environ.get("QWEN38_SERVING_LAYERS")
        indices = [int(i) for i in indices.split(",")] if indices else None
        obj = cls()
        obj.model = Model(str(checkpoint), mesh_device, layer_indices=indices)
        obj.checkpoint, obj.mesh = str(checkpoint), mesh_device
        obj.batch_size, obj.context_length = max_batch_size, max_seq_len
        if max_seq_len != obj.model.config.max_position_embeddings:
            raise ValueError("Serving must preserve the context contract")
        obj.generator = None
        obj.kv_cache = None
        obj.request_seeds = [secrets.randbelow(2**31) for _ in range(max_batch_size)]
        obj.allow_host_sampling = os.environ.get('QWEN38_HOST_SAMPLING') == '1'
        obj.host_logits_reads = 0
        obj.slot_ends = [None]*max_batch_size
        obj.state_pool = None
        obj.scheduler_block_size = 32
        obj.event_path = os.environ.get('QWEN38_SERVING_EVENTS')
        obj.trace_retirements = 0
        obj.decode_slots = set()
        obj.tt_supported_decode_batch_sizes = [max_batch_size]
        print(f"AUTOPORT_ADAPTER {__file__}:{cls.__name__} layers={obj.model.layer_indices} "
              f"precision={obj.model.precision_config}", flush=True)
        if os.environ.get('QWEN38_BENCHMARK_PHASES'):
            from .benchmark_phases import install
            install(obj)
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len, max_num_seqs, **kwargs):
        # The TT worker computes and caches its own pool before replacing the
        # CLI override. Read the original opt-in inside its current-config scope.
        from vllm.config import get_current_vllm_config_or_none
        config = get_current_vllm_config_or_none()
        if (config is not None and max_model_len == 262144 and max_num_seqs == 32
                and kwargs.get('num_devices') == 4 and kwargs.get('tt_data_parallel') == 1
                and config.cache_config.block_size == 2048
                and config.cache_config.mamba_block_size == 2048
                and config.cache_config.mamba_cache_mode == 'align'
                and config.cache_config.num_gpu_blocks_override == 385
                and config.speculative_config is None):
            # 160 token +32 output +3*2*32 recurrent +1 null =385 IDs.
            # This only changes capacity; model context and precision stay fixed.
            return max_model_len + 65536
        # Worker adds per-request recurrent-group IDs and output reservations.
        return max_model_len

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from .vllm_cache_specs import get_kv_cache_spec
        return get_kv_cache_spec(vllm_config)

    def allocate_hybrid_kv_cache(self, config):
        from .serving_state import ServingStatePool
        from .cache_block_config import scheduler_block_size
        block_size = scheduler_block_size()
        if any(group.kv_cache_spec.block_size != block_size for group in config.kv_cache_groups):
            raise ValueError("Resolved hybrid cache groups disagree with scheduler block setting")
        self.scheduler_block_size=block_size
        self.cache_groups={}
        for group_index,group in enumerate(config.kv_cache_groups):
            for name in group.layer_names:
                self.cache_groups[int(name.split('layers.')[1].split('.')[0])]=group_index
        states=self.allocate_kv_cache((config.num_blocks*(block_size//32),1,32,256),
                                       torch.bfloat16,len(self.model.layers))
        self.state_pool=ServingStatePool(self.model,states,
            num_blocks_by_layer={index:config.num_blocks
                for index,layer in zip(self.model.layer_indices,self.model.layers)
                if layer.layer_type=='linear_attention'})
        memory=ttnn.get_memory_view(self.mesh,ttnn.BufferType.DRAM)
        self.allocation_evidence=dict(
            scheduler_blocks=config.num_blocks,scheduler_block_size=block_size,
            group_count=len(config.kv_cache_groups),
            recurrent_snapshot_bytes_per_device=self.state_pool.allocated_bytes_per_device,
            attention_bytes_per_device=sum(2*int(state.key.shape[0])*8*1088
                for state in states if hasattr(state,'key')),
            active_recurrent_bytes_per_device=len(self.state_pool.pools)*self.batch_size*801792,
            allocated_bytes_per_device=memory.total_bytes_allocated_per_bank*memory.num_banks,
            free_bytes_per_device=memory.total_bytes_free_per_bank*memory.num_banks,
            basis='Physical selected BFP8 attention, FP32 recurrence, BF16 convolution; allocator checkpoint, not peak',
            selected_layers=self.model.layer_indices,
            precision_config=self.model.precision_config)
        return states

    def allocate_kv_cache(self, shape, dtype, num_layers):
        if shape[1:] != (1, 32, 256):
            raise ValueError(f"Expected TP4 attention pages [N,1,32,256], got {shape}")
        states = []
        for layer in self.model.layers:
            if layer.layer_type == "linear_attention":
                states.append(layer.allocate_state(batch_size=self.batch_size))
            else:
                def allocate():
                    return ttnn.zeros(shape, dtype=getattr(ttnn, layer.policy["kv_dtype"]),
                                      layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
                states.append(PagedState(allocate(), allocate()))
        self.kv_cache = states
        width = ((self.context_length + 255) // 256) * 8
        self.generator = Generator(self.model, self.checkpoint,
            batch_size=self.batch_size, context_length=self.context_length,
            kv_cache=states, page_table=torch.zeros(self.batch_size, width, dtype=torch.int32),
            # B32 prefix snapshots leave insufficient scratch for4096 rows;
            # measured OOM at1280. Internal chunks preserve logical context.
            serving_state=True,prefill_chunk_size=1024 if self.batch_size>16 else 4096)
        return states

    def warmup_model_prefill(self, *, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        started=time.perf_counter()
        first=self.generator.model_trace is None
        if self.state_pool is not None and self.generator.model_trace is None:
            self.state_pool.warmup()
        self.generator.prepare_decode()
        if first:
            self._event('prepared',duration_s=time.perf_counter()-started,
                execution_signatures={
                    **{f'decode:B{self.batch_size}:masked={masked}':str(trace)
                       for masked,trace in self.generator.decode_traces.items()},
                    f'sampling:B{self.batch_size}:{self.generator.sampling_strategy}':str(self.generator.sample_trace),
                    f'sampling_buffered:B{self.batch_size}:{self.generator.sampling_strategy}':str(self.generator.buffered_sample_trace)},
                prefill='eager; physical buckets prepared before capture',
                prepared_buckets=sorted(self.generator.prepared_buckets),
                physical_prefill_chunk_size=self.generator.prefill_chunk_size,
                preparation_events=self.generator.preparation_events,
                throw_exception_on_fallback=bool(ttnn.CONFIG.throw_exception_on_fallback),
                allocation=getattr(self,'allocation_evidence',None),
                preparation_memory=self.generator.last_preparation_memory)

    def warmup_model_decode(self, *, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        self.generator.prepare_decode()

    def _check_cache(self, cache):
        if cache is not self.kv_cache or self.generator is None:
            raise ValueError("Use the exact cache returned to vLLM by this allocator")

    def _event(self,kind,**fields):
        path=getattr(self,'event_path',None)
        if not path:
            return
        event=dict(event=kind,time=time.time(),pid=os.getpid(),batch=self.batch_size,
            counters=dict(self.generator.counters),host_logits_reads=self.host_logits_reads,
            retirement_count=self.trace_retirements,
            **fields)
        if self.state_pool is not None:
            event['state_pool']=dict(self.state_pool.counters)
        with open(path,'a') as output:
            output.write(json.dumps(event)+'\n')

    def _page_table(self, table):
        if self.scheduler_block_size != 32:
            count=self.scheduler_block_size//32
            table=(table.to(torch.int32).unsqueeze(-1)*count+
                   torch.arange(count,dtype=torch.int32)).flatten(1)
        result = torch.zeros_like(self.generator.page_table_host)
        if table.shape[0] > result.shape[0] or table.shape[1] > result.shape[1]:
            raise ValueError("Scheduler page table exceeds the configured capacity")
        result[:table.shape[0], :table.shape[1]] = table
        return result

    def _attention_table(self, table, per_layer):
        if self.state_pool is None:
            return table
        if per_layer is None:
            raise ValueError('Hybrid serving requires per-layer scheduler tables')
        index=next(i for i,layer in zip(self.model.layer_indices,self.model.layers)
                   if layer.layer_type=='full_attention')
        return per_layer[index]

    def _snapshot_ids(self, tables, row, end):
        if end<=0 or end%self.scheduler_block_size:
            raise ValueError('Prefix snapshots require an exact scheduler block boundary')
        column=(end-1)//self.scheduler_block_size
        ids={index:int(tables[index][row,column]) for index in self.state_pool.pools}
        if any(index<=0 for index in ids.values()):
            raise ValueError('A live recurrent checkpoint cannot use the null block')
        return ids

    def release_request(self, slot):
        self._event('release_request',slot=int(slot),end=self.slot_ends[slot],retirements=0)
        self.slot_ends[slot]=None
        self.decode_slots.discard(slot)
        self.request_seeds[slot]=secrets.randbelow(2**31)

    def _save_decode_boundary(self,start_pos,reload_inputs,layer_tables):
        if self.state_pool is None:
            return
        active=self.generator._decode_active.tolist()
        for slot,live in enumerate(active):
            if not live:
                continue
            if reload_inputs:
                self.slot_ends[slot]=int(start_pos[slot])+1
            else:
                self.slot_ends[slot]+=1
            end=self.slot_ends[slot]
            if end is not None and end%self.scheduler_block_size==0:
                self.state_pool.save(slot,self._snapshot_ids(layer_tables,slot,end))

    def _sampling(self, params, slots=None, *, reset_state=False, offsets=0, reset_slots=None):
        if params is None:
            raise ValueError("Host sampling is not enabled for this adapter")
        slots = list(range(self.batch_size)) if slots is None else slots
        k, p, temp = ([1]*self.batch_size, [0.0]*self.batch_size,
                      [1.0]*self.batch_size)
        for row, slot in enumerate(slots):
            greedy = params.temperature[row] == 0
            k[slot] = 1 if greedy else params.top_k[row]
            p[slot] = 0.0 if greedy else params.top_p[row]
            temp[slot] = 1.0 if greedy else params.temperature[row]
            if params.seed[row] is not None:
                self.request_seeds[slot] = params.seed[row]
        self.generator.set_sampling_params(top_k=k, top_p=p, temperature=temp, reset_seed=False)
        if reset_state:
            self.generator.reset_sampling_state(seed=self.request_seeds,offsets=offsets,slots=reset_slots)

    def prefill_forward(self, tokens, start_pos, prompt_lens, page_table, kv_cache,
                        empty_slots=None, sampling_params=None, intermediate_prefill_mask=None,
                        **kwargs):
        self._check_cache(kv_cache)
        started=time.perf_counter()
        slots = list(range(len(prompt_lens))) if empty_slots is None else list(empty_slots)
        gen = self.generator
        layer_tables=kwargs.get('page_tables_per_layer')
        attention_table=self._attention_table(page_table,layer_tables)
        attention_table=self._page_table(attention_table)
        intermediate=(torch.zeros(len(slots),dtype=torch.bool) if intermediate_prefill_mask is None
                      else torch.as_tensor(intermediate_prefill_mask,dtype=torch.bool).reshape(-1))
        if len(intermediate)!=len(slots):
            raise ValueError('Intermediate prefill mask requires one value per submitted row')
        all_intermediate=bool(intermediate.all())
        host_mode = sampling_params is None
        if host_mode and not all_intermediate and not self.allow_host_sampling:
            raise ValueError('Shared-test host sampling requires explicit QWEN38_HOST_SAMPLING=1')
        results = (torch.zeros(len(slots),1,self.model.config.vocab_size) if host_mode and not all_intermediate
                   else torch.zeros(len(slots),1,dtype=torch.int64))
        groups=sorted({(int(p),int(end)-int(p),bool(intermediate[i]))
                       for i,(p,end) in enumerate(zip(start_pos,prompt_lens))})
        self._event('prefill_submit',starts=[int(p) for p in start_pos],
            ends=[int(p) for p in prompt_lens],slots=[int(s) for s in slots],
            active_decode_slots=sorted(self.decode_slots-set(slots)))
        self.decode_slots.difference_update(slots)
        for start, length, suppressed in groups:
            rows = [i for i,p in enumerate(start_pos)
                    if int(p)==start and int(prompt_lens[i])-start==length
                    and bool(intermediate[i])==suppressed]
            lengths = [0]*self.batch_size
            for row in rows:
                lengths[slots[row]] = int(prompt_lens[row])-start
            if start == 0:
                gen.reset_state_slots([slots[row] for row in rows])
                for row in rows:
                    self.request_seeds[slots[row]]=secrets.randbelow(2**31)
            elif self.state_pool is not None:
                for row in rows:
                    slot=slots[row]
                    if self.slot_ends[slot] != start:
                        self.state_pool.restore(slot,self._snapshot_ids(layer_tables,row,start))
            table = gen.page_table_host.clone()
            values = torch.zeros(self.batch_size,max(lengths),dtype=torch.int64)
            for row in rows:
                slot=slots[row]
                table[slot].zero_()
                table[slot]=attention_table[row]
                values[slot,:lengths[slot]]=tokens[row,start:int(prompt_lens[row])]
            consumed=0
            while consumed<length:
                absolute=start+consumed
                piece=min(length-consumed,gen.prefill_chunk_size,
                          self.scheduler_block_size-absolute%self.scheduler_block_size
                          if self.state_pool is not None else length)
                logits=gen.prefill_continuation(values[:,consumed:consumed+piece],
                    page_table=table,kv_cache=kv_cache,
                    prompt_lens=[piece if n else 0 for n in lengths],start_position=absolute,
                    compute_logits=not suppressed and consumed+piece==length)
                consumed+=piece
                end=start+consumed
                self._event('prefill_chunk',start=absolute,length=piece,
                    slots=[int(slots[row]) for row in rows],
                    compute_logits=not suppressed and consumed==length)
                if self.state_pool is not None and end%self.scheduler_block_size==0:
                    for row in rows:
                        column=(end-1)//self.scheduler_block_size
                        allocated=[int(layer_tables[i][row,column]) for i in self.state_pool.pools]
                        if any(allocated):
                            self.state_pool.save(slots[row],self._snapshot_ids(layer_tables,row,end))
            for row in rows:
                self.slot_ends[slots[row]]=int(prompt_lens[row])
            if suppressed:
                continue
            if host_mode:
                host=gen.host_logits(logits)
                self.host_logits_reads+=1
            else:
                offsets=[0]*self.batch_size
                for row in rows:
                    offsets[slots[row]]=int(prompt_lens[row])
                active_slots=[slots[row] for row in rows]
                self._sampling(sampling_params,slots,reset_state=True,offsets=offsets,
                               reset_slots=active_slots)
                output=gen.sample_prefill(logits,active_slots=active_slots)
                host=self.process_decode_output_host(self.read_decode_output(output),is_tokens=True)
            for row in rows:
                results[row]=host[slots[row]]
        # Text MRoPE uses identical position axes, hence zero per-request delta.
        self._event('prefill_complete',duration_s=time.perf_counter()-started,retirements=0)
        return results, torch.zeros(len(slots),dtype=torch.int64)

    def decode_forward(self, tokens, start_pos, page_table, kv_cache,
                       sampling_params=None, read_from_device=True, slot_remap=None,
                       reload_inputs=True, reload_page_table=False,
                       reload_sampling_params=False, reset_sampling_state=False, **kwargs):
        if 'reset_batch' in kwargs:
            raise TypeError('The adapter requires decode input update contract version1')
        if reload_inputs and reload_page_table:
            raise ValueError('Input reload already includes page tables')
        if reset_sampling_state and not reload_inputs:
            raise ValueError('Sampling state reset requires authoritative inputs')
        deltas=kwargs.get('rope_deltas_all_users')
        if deltas is not None and torch.as_tensor(deltas).count_nonzero():
            raise ValueError('This text-only adapter requires zero MRoPE deltas')
        self._check_cache(kv_cache)
        if slot_remap is not None:
            self.generator.remap_state_slots(slot_remap)
            self.request_seeds=[self.request_seeds[int(i)] for i in slot_remap]
            self.slot_ends=[self.slot_ends[int(i)] for i in slot_remap]
            self.decode_slots={i for i,old in enumerate(slot_remap) if int(old) in self.decode_slots}
        layer_tables=kwargs.get('page_tables_per_layer')
        table=self._attention_table(page_table,layer_tables)
        scheduler_table=table if self.event_path else None
        if sampling_params is None:
            if not self.allow_host_sampling:
                raise ValueError('Shared-test host sampling requires explicit QWEN38_HOST_SAMPLING=1')
            if not reload_inputs:
                raise ValueError('Host sampling requires authoritative scheduler inputs')
            self.host_logits_reads+=1
            host=self.generator.decode_forward(tokens,start_pos,page_table=self._page_table(table),
                kv_cache=kv_cache,return_logits=True).reshape(self.batch_size,1,-1)
            self._save_decode_boundary(start_pos,True,layer_tables)
            return host
        if reload_sampling_params or reset_sampling_state:
            # Counter indexed by absolute sampled-token position; re-prefill and
            # preemption can reconstruct it without replaying host token feedback.
            offsets=torch.where(start_pos>=0,start_pos+1,0) if reset_sampling_state else 0
            self._sampling(sampling_params,reset_state=reset_sampling_state,offsets=offsets)
        table=(self._page_table(table) if reload_inputs or reload_page_table
               else self.generator.page_table)
        output=self.generator.decode_forward(tokens,start_pos,page_table=table,
            kv_cache=kv_cache,return_device=True,reload_inputs=reload_inputs,
            reload_page_table=reload_page_table)
        self._save_decode_boundary(start_pos,reload_inputs,layer_tables)
        if reload_inputs:
            self.decode_slots={i for i,live in enumerate(self.generator._decode_active.tolist()) if live}
        if self.event_path:
            active_slots=sorted(self.decode_slots)
            self._event('decode_submit',reload_inputs=reload_inputs,reload_page_table=reload_page_table,
                trace_id=str(self.generator.model_trace),retirements=0,
                active_slots=active_slots,slot_ends=list(self.slot_ends),
                slot_remap=None if slot_remap is None else [int(i) for i in slot_remap],
                scheduler_table_rows={str(i):scheduler_table[i].tolist() for i in active_slots})
        if not read_from_device:
            return output
        return self.process_decode_output_host(self.read_decode_output(output),is_tokens=True)

    def read_decode_output(self, output, async_read=False):
        if isinstance(output,torch.Tensor):
            return (output,[]) if async_read else output
        # Only the replicated sampled IDs cross the host boundary.
        host=ttnn.get_device_tensors(output)[0].cpu(blocking=not async_read)
        self.generator.counters['readbacks']+=1
        return (host,[ttnn.record_event(self.mesh,0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=False):
        if not is_tokens:
            raise ValueError("This path returns sampled tokens, not full logits")
        if not isinstance(output,torch.Tensor):
            # Synchronous plugin finalization may hand us the raw replicated
            # device output; asynchronous finalization already performed this read.
            if output.storage_type()!=ttnn.StorageType.HOST:
                output=self.read_decode_output(output)
            output=ttnn.to_torch(output)
        return output.reshape(self.batch_size,1).long()

    def release_persistent_capture(self):
        if self.generator is None:
            return
        gen = self.generator
        ttnn.synchronize_device(self.mesh)
        traces = set(gen.decode_traces.values())
        traces.update(t for t in (gen.sample_trace, gen.buffered_sample_trace) if t is not None)
        if traces:
            print('AUTOPORT_SHUTDOWN_AUDIT '+json.dumps(dict(
                host_sampling_compatibility=self.allow_host_sampling,
                host_logits_reads=self.host_logits_reads,counters=dict(gen.counters),
                layer_indices=self.model.layer_indices,batch=self.batch_size,
                trace_ids=sorted(str(t) for t in traces))),flush=True)
        for trace in traces:
            ttnn.release_trace(self.mesh, trace)
            self.trace_retirements+=1
        self._event('shutdown_release',released_trace_ids=[str(t) for t in traces],
                    retirements=len(traces))
        gen.decode_traces.clear()
        gen.model_trace = gen.sample_trace = gen.buffered_sample_trace = None
