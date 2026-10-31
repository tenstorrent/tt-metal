# APC M0' (DSV41_APC_GRID=512) merge notes

Branch ssinghal/dsv4p1-apc-m0p (off ssinghal/dsv4p1-b8-final 95a82b1cadf). Plugin: NO change needed (`prefix_cache_hit_policy` = `all` is already accepted by `round_tt_prefix_hit`; the model advertises it when the grid is on).

New files (no conflicts): `tt/apc_grid.py` (pure policy `GridPolicy.plan(tokens, total) -> {B: kind}`; reusable by the device-resident tier as the fallback policy), `tests/test_apc_grid.py`, `tools/apc_paged_sim.py` (byte copy of the design branch file, identical content merges cleanly).

Hunks in shared files (all small, additive):

| File | Where | Change | Conflict risk with apc_img2 |
|---|---|---|---|
| tt/prefix_cache.py | `PrefixCache.__init__`, after `self.stats = dict(...)` | `self.grid`, `self.plans`, `self.serial`, GridPolicy creation from `DSV41_APC_GRID` | none (img2 does not touch `__init__`) |
| tt/prefix_cache.py | new methods `new_prompt`, `wants_user`, `_thin` before `_has_image` | additive | adjacent to img2's new `_keyed` / `images_enabled` block (+35 lines at the same place): keep both, order irrelevant |
| tt/prefix_cache.py | `begin_snapshot` first line | `self.wants(B)` -> `self.wants_user(b, B)` | textual: img2 edits the signature (`vs=None`) and the next lines. Resolve: keep img2's body, apply this one-token change |
| tt/prefix_cache.py | `begin_snapshot`, before `pend.append(...)` | two lines `p.kind = ...`, `p.serial = ...` | adjacent to img2's `if kt is not tokens: p.snap.ktoks = ...`: keep both |
| tt/prefix_cache.py | `finish_pending` after `snap.full = full` | `snap.kind, snap.serial = ...` and `self._thin(snap)` after `_store` | none |
| tt/prefix_cache.py | `_thin` | uses `snap.ktoks` when present (img2's key tokens) so the lineage test uses the stored key | already compatible |
| tt/dsv41_model.py | `prefill_interleaved_one`, between `t_apc += apc.finish_pending()` and `new_ = self._apc_restore_wave(` | `if apc.grid is not None:` loop calling `apc.new_prompt(b, t, e or prefill_total)` for the rows with `st == 0` and no image | textual: img2 rewrites the lines just below (`cand`, `img_cont`). Resolve: keep both; the loop should iterate `cand` rows with `st == 0`. Image prompts get NO plan (`new_prompt` skips them via `_has_image`; with img2 use its keyed tokens to plan image prompts too: pass `kt`) |
| tt/dsv41_model.py | `warm_apc`, after `sizes = [...]` | with the grid on, also warm every multiple of the chunk up to `DSV41_WARM_APC_GRID_MAX` (32768) | none |
| tt/generator_vllm.py | `_chunked_prefill_capabilities`, `prefix_cache_hit_policy` | default `all` instead of `pow2` when `DSV41_APC_GRID` is set | none (img2 hunks are at 883, 1052, 1177) |

Not touched: `window_needs_sync` (the grid needs `DSV41_APC_DEFER=1`, the default: the plan is consulted only in `begin_snapshot`, so no extra host sync is needed), the lookup (`PrefixStore.lookup` already scans every chunk boundary), restore.
