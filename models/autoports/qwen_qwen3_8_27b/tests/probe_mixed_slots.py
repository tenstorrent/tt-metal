# SPDX-License-Identifier: Apache-2.0
"""Reduced B8 mixed/fixed-slot isolation probe. Run with trace allocation tracking.

Uses real layers0/3 and the complete real terminal path. The later nonzero-state
prefill check tests slot isolation only, not active-row continuation semantics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator, build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='.cache/qwen38-probe')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', default='models/autoports/qwen_qwen3_8_27b/doc/full_model/mixed_slots.json')
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError('At least two generation steps are required')
    if os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING') != '1' or os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE', '0') != '0':
        raise RuntimeError('Run with TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0')
    torch.set_num_threads(8)
    started = time.monotonic()
    result = {'scope': 'reduced real layers0/3 plus real terminal, B8/context512',
              'prepared_buckets': [1,64,96,128,192,256,320,384,512], 'status': 'running', 'checks': {},
              'trace_allocation_tracking': True, 'program_cache_allocations_included': True}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    result['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), *[source_root/'tt'/name for name in
                                ('generator.py', 'model.py', 'multichip_decoder.py', 'optimized_decoder.py')]]}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        print('MIXED_BUILD', flush=True)
        initial = build_generator(source_root, mesh, checkpoint=args.checkpoint,
                                  layer_indices=[0, 3], batch_size=8, context_length=512, prepare=False)
        # A caller binds already allocated cache and page-table ownership before
        # preparation. Both generators' buffers predate every captured trace;
        # only the externally bound generator is subsequently executed.
        external_cache = initial.kv_cache
        external_map = initial.page_table_host.clone()
        try:
            Generator(initial.model,args.checkpoint,batch_size=8,context_length=65,
                kv_cache=external_cache,page_table=torch.arange(24,dtype=torch.int32).reshape(8,3))
        except ValueError as error:
            assert 'multiple of8' in str(error)
        else:
            raise AssertionError('Unaligned physical page map was accepted')
        result['checks']['external_physical_page_map_alignment_validated']=True
        gen = Generator(initial.model, args.checkpoint, batch_size=8, context_length=512,
                        kv_cache=external_cache, page_table=initial.page_table)
        assert not gen.owns_cache and gen.kv_cache is external_cache
        assert not gen.owns_page_table and gen.page_table is initial.page_table
        print('MIXED_SEED_EXTERNAL_CACHE', flush=True)
        initial.prefill_forward(torch.full((8,33),101,dtype=torch.int64),
            page_table=initial.page_table,kv_cache=external_cache,prompt_lens=[33]*8,return_device=True)
        saved_external = [[ttnn.to_torch(local).clone()
                           for value in vars(state).values() for local in ttnn.get_device_tensors(value)]
                          for state in external_cache]
        assert any(torch.count_nonzero(value).item() for layer in saved_external for value in layer)
        print('MIXED_PREPARE_FAILURE_RESTORE', flush=True)
        original_decode = gen._decode
        def failed_warmup(*args_,**kwargs_):
            original_decode(*args_,**kwargs_)
            raise RuntimeError('injected warmup failure after cache mutation')
        gen._decode = failed_warmup
        try:
            gen.prepare_decode(prefill_lengths=(1,33,65,128,161,256,289,384,511))
        except RuntimeError as error:
            assert str(error) == 'injected warmup failure after cache mutation'
        else:
            raise AssertionError('Injected preparation failure did not propagate')
        finally:
            gen._decode = original_decode
        for state,saved in zip(external_cache,saved_external):
            current = [ttnn.to_torch(local) for value in vars(state).values()
                       for local in ttnn.get_device_tensors(value)]
            assert all(torch.equal(old,new) for old,new in zip(saved,current))
        assert gen.model_trace is None and not gen._warming and gen._warm_layer_indices is None
        result['checks']['failed_prepare_restores_nonzero_external_cache_all_devices'] = True
        print('MIXED_PREPARE', flush=True)
        gen.prepare_decode(prefill_lengths=(1,33,65,128,161,256,289,384,511))
        for state,saved in zip(external_cache,saved_external):
            current = [ttnn.to_torch(local) for value in vars(state).values()
                       for local in ttnn.get_device_tensors(value)]
            assert all(torch.equal(old,new) for old,new in zip(saved,current))
        del saved_external
        result['checks']['prepare_restores_nonzero_external_cache_all_devices'] = True
        assert gen.prepared_buckets == set(result['prepared_buckets'])
        captures = gen.counters['captures']
        assert captures == 4, gen.counters

        def trace_ids():
            return {**{str(mode): str(trace) for mode, trace in gen.decode_traces.items()},
                    'sampling': str(gen.sample_trace), 'buffered_sampling': str(gen.buffered_sample_trace)}

        identities = trace_ids()
        cache_addresses = [[value.buffer_address() for value in vars(state).values()] for state in external_cache]

        def host_rows(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1).tolist()

        def snapshot():
            # Host-only snapshots do not allocate device storage after capture.
            return [{name: [ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(value)]
                     for name, value in vars(state).items()} for state in external_cache]

        def unchanged(before, after, slots, label):
            pages = gen.page_table_host[list(slots)].reshape(-1).long()
            for layer, (left, right) in enumerate(zip(before, after)):
                for name in left:
                    rows = pages if name in ('key', 'value') else list(slots)
                    assert len(left[name]) == len(right[name]) == 4
                    for rank, (old, new) in enumerate(zip(left[name], right[name])):
                        assert torch.equal(old[rows], new[rows]), (label, layer, name, rank, list(slots))
            result['checks'][label] = {'inactive_slots': list(slots), 'all_four_devices_exact': True}

        def reused(label):
            assert trace_ids() == identities, label
            assert gen.counters['captures'] == captures, label
            assert cache_addresses == [[value.buffer_address() for value in vars(state).values()]
                                       for state in external_cache], label
            result['checks'][label] = True

        print('MIXED_ALL_ACTIVE_FIRST', flush=True)
        all_prompts = [[100 + row] * 33 for row in range(8)]
        all_first = gen.generate(all_prompts, args.steps)
        assert len(all_first) == 8 and all(len(row) == args.steps for row in all_first)
        assert host_rows(gen.positions) == [33 + args.steps - 1] * 8
        assert gen.model_trace == gen.decode_traces[False]
        reused('all_active_first_trace_reuse')

        print('MIXED_GENERATE', flush=True)
        mixed_prompts = [[110], [111] * 33, [112] * 35, [113] * 65]
        mixed = gen.generate(mixed_prompts, args.steps)
        assert len(mixed) == 4 and all(len(row) == args.steps for row in mixed)
        assert host_rows(gen.positions) == [n + args.steps - 1 for n in (1, 33, 35, 65)] + [-1] * 4
        assert gen.model_trace == gen.decode_traces[True]
        mixed_state = snapshot()
        zero_state = [{name: [torch.zeros_like(local) for local in locals_] for name, locals_ in layer.items()}
                      for layer in mixed_state]
        unchanged(zero_state, mixed_state, range(4, 8), 'mixed_generation_inactive_zero_state')
        reused('mixed_trace_reuse')
        mixed_again = gen.generate(mixed_prompts, args.steps)
        assert mixed == mixed_again
        result['checks']['mixed_outputs_repeat_exactly'] = True
        reused('mixed_repeat_trace_reuse')

        print('MIXED_ALL_ACTIVE_RETURN', flush=True)
        all_second = gen.generate(all_prompts, args.steps)
        assert all_first == all_second
        assert gen.model_trace == gen.decode_traces[False]
        result['checks']['all_active_outputs_repeat_exactly'] = True
        reused('all_active_return_trace_reuse')

        # Every slot now has a real, nonzero state. Preserve unselected rows while
        # the scheduler advances selected rows through the public low-level API.
        before = snapshot()
        for layer in before:
            if 'recurrent' in layer:
                assert all(torch.count_nonzero(local[row]).item() > 0 for local in layer['recurrent'] for row in range(8))
        print('MIXED_LOW_LEVEL_DECODE', flush=True)
        active = [0, 2, 5]
        positions = torch.full((8,), 33 + args.steps - 1, dtype=torch.int32)
        tokens = torch.tensor([[row[-1]] for row in all_second], dtype=torch.int64)
        sampled = gen.decode_forward(tokens, positions, page_table=external_map, kv_cache=external_cache,
                                     active_slots=active, return_logits=False)
        assert sampled.shape == (8,)
        inactive = [row for row in range(8) if row not in active]
        assert sampled[inactive].tolist() == [0] * len(inactive)
        assert host_rows(gen.positions) == [33 + args.steps if row in active else -1 for row in range(8)]
        unchanged(before, snapshot(), inactive, 'masked_decode_preserves_nonzero_inactive_state')
        reused('low_level_external_cache_trace_reuse')

        print('MIXED_PREFILL_GROUP_ISOLATION', flush=True)
        before = snapshot()
        prefill_lens = [0, 1, 0, 33, 0, 0, 65, 0]
        padded = torch.zeros(8, 65, dtype=torch.int64)
        for row, length in enumerate(prefill_lens):
            if length:
                padded[row, :length] = 120 + row
        # Intentional nonzero initial state: this checks untouched slots, not the
        # numerical meaning of the selected slots' newly overwritten prefixes.
        logits = gen.prefill_forward(padded, page_table=external_map, kv_cache=external_cache,
                                     prompt_lens=prefill_lens, return_all_logits=False, return_device=True)
        assert logits is gen.prefill_output
        del logits
        untouched = [row for row, length in enumerate(prefill_lens) if not length]
        unchanged(before, snapshot(), untouched, 'prefill_groups_preserve_nonzero_inactive_state')
        # Replay after eager prefill also exercises allocation safety in this
        # mixed group ordering; only selected rows are meaningful outputs here.
        gen.decode_forward(torch.full((8, 1), 121), torch.tensor(prefill_lens, dtype=torch.int32),
                           page_table=external_map, kv_cache=external_cache,
                           active_slots=[1, 3, 6], return_logits=False)
        reused('prefill_then_decode_trace_reuse')

        print('MIXED_DEVICE_PAGE_MAP', flush=True)
        gen.reset()
        changed_map = external_map.roll(1,dims=0)
        # Deliberately bypass refresh_page_table: the bound device map is authoritative.
        gen.refresh(changed_map,gen.page_table)
        gen.prefill_forward(torch.full((8,33),101,dtype=torch.int64),
            page_table=gen.page_table,kv_cache=external_cache,prompt_lens=[33]*8,return_device=True)
        for local in ttnn.get_device_tensors(gen.buckets[64]['fill_page_table']):
            assert torch.equal(ttnn.to_torch(local),changed_map[:,:8])
        remapped_state = snapshot()
        gen.refresh_page_table(external_map)
        gen.reset()
        gen.prefill_forward(torch.full((8,33),101,dtype=torch.int64),
            page_table=external_map,kv_cache=external_cache,prompt_lens=[33]*8,return_device=True)
        identity_state = snapshot()
        for remapped,identity in zip(remapped_state,identity_state):
            for name in ('key','value'):
                if name not in identity:
                    continue
                for actual,expected in zip(remapped[name],identity[name]):
                    assert torch.equal(actual[changed_map.flatten().long()],
                                       expected[external_map.flatten().long()])
        result['checks']['inplace_device_page_map_prefill_targets_exact_pages'] = True
        gen.prefill_forward(torch.full((8,33),102,dtype=torch.int64),
            page_table=external_map,kv_cache=external_cache,prompt_lens=[33]*8,
            start_position=256,return_device=True)
        for local in ttnn.get_device_tensors(gen.buckets[64]['fill_page_table']):
            assert torch.equal(ttnn.to_torch(local),external_map[:,8:16])
        positions = ttnn.to_torch(ttnn.get_device_tensors(gen.buckets[64]['positions'])[0])
        assert torch.equal(positions.long(),torch.arange(256,320).expand(8,-1))
        result['checks']['continuation_uses_absolute_positions_and_runtime_page_offset'] = True
        print('MIXED_CONTINUATION_NUMERICS',flush=True)
        def pcc(a,b):
            a=a.float().reshape(-1);b=b.float().reshape(-1)
            return 1.0 if torch.equal(a,b) else float(torch.corrcoef(torch.stack([a,b]))[0,1])
        continuation=[]
        for prefix, tail in ((128,33),(256,33),(384,127)):
            total=prefix+tail
            complete_tokens=torch.full((8,total),103,dtype=torch.int64)
            gen.reset()
            whole=gen.prefill_forward(complete_tokens,page_table=external_map,kv_cache=external_cache,
                prompt_lens=[total]*8)
            whole_state=snapshot()
            gen.reset()
            gen.prefill_forward(complete_tokens[:,:prefix],page_table=external_map,kv_cache=external_cache,
                prompt_lens=[prefix]*8,return_device=True)
            split=gen.prefill_forward(complete_tokens[:,prefix:],page_table=external_map,kv_cache=external_cache,
                prompt_lens=[tail]*8,start_position=prefix)
            correlation=pcc(whole,split)
            assert correlation>=.995,correlation
            state_correlations=[]
            for old,new in zip(whole_state,snapshot()):
                for name in ('conv','recurrent'):
                    if name in old:
                        for a_,b_ in zip(old[name],new[name]):
                            score=pcc(a_,b_);state_correlations.append(score)
                            assert score>=.995,(name,score)
            if prefix == 384:
                expected=external_map[:,torch.arange(12,20).clamp_max(15)]
                for local in ttnn.get_device_tensors(gen.buckets[128]['fill_page_table']):
                    assert torch.equal(ttnn.to_torch(local),expected)
                result['checks']['near_end_odd128_page_window_bounded'] = True
            continuation.append({'prefix':prefix,'tail':tail,'logits_pcc':correlation,'state_pcc':state_correlations})
        result['checks']['continuation_matches_whole_prefill']=continuation
        print('MIXED_LOGIT_REPRODUCIBILITY',flush=True)
        cases=[(101,33),(102,65),(101,33),(103,35),(104,33),(102,65),(103,35),(101,33)]
        def logit_request(order):
            gen.reset()
            lengths=[cases[i][1] for i in order]
            tokens=torch.zeros(8,max(lengths),dtype=torch.int64)
            for row,i in enumerate(order):
                tokens[row,:lengths[row]]=cases[i][0]
            pref=gen.prefill_forward(tokens,page_table=external_map,kv_cache=external_cache,
                prompt_lens=lengths).reshape(8,-1)
            dec=gen.decode_forward(torch.full((8,1),105,dtype=torch.int64),
                torch.tensor(lengths,dtype=torch.int32),page_table=external_map,
                kv_cache=external_cache,return_logits=True).reshape(8,-1)
            return pref,dec
        order=list(range(8));permutation=[7,3,5,0,6,2,4,1]
        first=logit_request(order);repeat=logit_request(order);permuted=logit_request(permutation)
        inverse=[permutation.index(i) for i in order]
        for phase,a_,b_,c_ in zip(('prefill','decode'),first,repeat,permuted):
            assert torch.equal(a_,b_),(phase,'repeat')
            assert torch.equal(a_,c_[inverse]),(phase,'slot permutation',float((a_-c_[inverse]).abs().max()))
            for left,right in ((0,2),(0,7),(1,5),(3,6)):
                assert torch.equal(a_[left],a_[right]),(phase,left,right)
        result['checks']['prefill_decode_logits_repeat_and_slot_permutation_exact']=True
        reused('logit_reproducibility_trace_reuse')
        before = snapshot()
        for logits_out in (False,True):
            for device_out in (False,True):
                empty = gen.decode_forward(torch.zeros(8,1,dtype=torch.int64),
                    torch.full((8,),-1,dtype=torch.int32),page_table=external_map,
                    kv_cache=external_cache,return_logits=logits_out,return_device=device_out)
                if device_out:
                    assert empty is (gen.logits if logits_out else gen.tokens)
                else:
                    assert empty.shape == ((8,gen.model.config.vocab_size) if logits_out else (8,))
                    assert not torch.count_nonzero(empty)
        unchanged(before,snapshot(),range(8),'all_inactive_decode_preserves_cache')
        reused('device_page_map_and_all_inactive_trace_reuse')

        result.update(status='pass', elapsed_seconds=time.monotonic() - started,
                      trace_ids=identities, captures=captures, counters=dict(gen.counters),
                      external_cache_bound=True, mixed_prompt_lengths=[1, 33, 35, 65],
                      mixed_outputs=mixed, all_active_outputs=all_first,
                      limitations=['Reduced model, not full-model quality/performance evidence.',
                                   'Explicit prepared bucket subset; does not prove whole-domain preparation.',
                                   'Nonzero-state prefill checks isolation only, not selected-row continuation correctness.'])
        print('MIXED_PASS', json.dumps(result['checks']), flush=True)
    except BaseException as error:
        result.update(status='fail', error_type=type(error).__name__, error=str(error)[:2000],
                      elapsed_seconds=time.monotonic() - started)
        print('MIXED_FAIL', type(error).__name__, str(error), flush=True)
        raise
    finally:
        target.write_text(json.dumps(result, indent=2) + '\n')
        ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
