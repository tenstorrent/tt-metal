# DeepSeek MoE test rezultati (Blackhole)

Datum: 2026-06-24
Branch: `pmilojevic/minimax-glm-tests`
Pokrenuto: `python_env/bin/python -m pytest <file> -p no:randomly`

## Zaključak

**Nema nijednog novog pada.** Svi testovi koji padaju na device-u su **već xfail-ovani**, i nijedan xfail nije neočekivano prošao (0 xpassed). Ne treba dodavati nove xfail markere.

| Fajl | passed | xfailed | failed | xpassed |
|---|---|---|---|---|
| test_deepseek_moe_post_combine_reduce.py | 85 | 5 | 0 | 0 |
| test_single_routed_expert.py | 22 | 6 | 0 | 0 |

## Testovi koji padaju (već xfail-ovani)

### test_deepseek_moe_post_combine_reduce.py

- `test_structured_data[dsv4_flash]` — structured PCC ispod thresholda — [#46609](https://github.com/tenstorrent/tt-metal/issues/46609)
- `test_structured_data[gptoss_120b]` — structured reshape invalid (tile_width hardcoded 1024) — [#46731](https://github.com/tenstorrent/tt-metal/issues/46731)
- `test_multi_chunk_structured[gptoss_120b-num_tokens=4096]` — [#46731](https://github.com/tenstorrent/tt-metal/issues/46731)
- `test_multi_chunk_structured[gptoss_120b-num_tokens=6400]` — [#46731](https://github.com/tenstorrent/tt-metal/issues/46731)
- `test_multi_chunk_structured[gptoss_120b-num_tokens=8192]` — [#46731](https://github.com/tenstorrent/tt-metal/issues/46731)

### test_single_routed_expert.py

- `test_single_routed_expert_models[...-dsv4_pro-25k]` — circular buffers prerastu L1 na velikom broju tokena — [#46608](https://github.com/tenstorrent/tt-metal/issues/46608)
- `test_single_routed_expert_faked_token_count_models[...-dsv4_pro-25k-alloc-4k-active]` — [#46608](https://github.com/tenstorrent/tt-metal/issues/46608)
- `test_single_routed_expert_models[...-gptoss_120b-1k]` — K_gate_tiles nije deljiv sa in0_block_w_gu — [#47604](https://github.com/tenstorrent/tt-metal/issues/47604)
- `test_single_routed_expert_faked_token_count_models[...-gptoss_120b-1k-alloc-0k-active]` — [#47604](https://github.com/tenstorrent/tt-metal/issues/47604)
- `test_single_routed_expert_models[...-gptoss_120b-25k]` — [#47604](https://github.com/tenstorrent/tt-metal/issues/47604)
- `test_single_routed_expert_faked_token_count_models[...-gptoss_120b-25k-alloc-4k-active]` — [#47604](https://github.com/tenstorrent/tt-metal/issues/47604)
