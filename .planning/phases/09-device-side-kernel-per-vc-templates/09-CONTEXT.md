# Phase 9: Device-side kernel per-VC templates - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Add per-VC template helpers to device kernel headers and refactor `fabric_erisc_router.cpp` to use templated per-VC functions. Replace all flat cross-VC sender channel arrays with per-VC scoped access in both constexpr configuration and runtime state. CT args wire format must remain unchanged.

</domain>

<decisions>
## Implementation Decisions

### Template helper design
- Create `constexpr std::array<size_t, MAX_NUM_VCS> MAX_NUM_SENDER_CHANNELS_PER_VC = {MAX_NUM_SENDER_CHANNELS_VC0, MAX_NUM_SENDER_CHANNELS_VC1}` — sized to `MAX_NUM_VCS`, not hardcoded 2
- Per-VC accessors use function templates with both VC and channel as template args: `template<size_t VC, size_t CH> constexpr bool is_sender_channel_serviced_vc()`
- Separate per-VC constexpr arrays built from flat arrays at compile time (Option 3 from discussion) — not view/offset-based access into flat arrays

### Router function refactoring scope
- ALL functions that iterate over flat `NUM_SENDER_CHANNELS` or `MAX_NUM_SENDER_CHANNELS` with `is_sender_channel_serviced` guards get templated on VC — not just `any_sender_channels_active` and `update_telemetry`
- This includes: `any_sender_channels_active`, `update_telemetry`, sender channel init loop (~line 1733), and all ~8 loops that iterate `NUM_SENDER_CHANNELS` in `fabric_erisc_router.cpp`
- Call sites use `tt::stl::concepts::for_each_index<MAX_NUM_VCS>([&]<size_t VC>{ func<VC>(...); })` for constexpr dispatch over VCs

### Runtime array splitting
- Flat runtime arrays (`local_sender_channel_free_slots_stream_ids`, `channel_connection_established`, `sender_channel_from_receiver_credits`) split into per-VC using `std::tuple<std::array<T, VC0_count>, std::array<T, VC1_count>>` with `std::get<VC>()` access
- `std::tuple` is already used in device headers (`edm_fabric_flow_control_helpers.hpp`, `fabric_erisc_datamover_channels.hpp`) — confirmed available on RISC-V toolchain
- Each VC gets exactly-sized storage — no padding/wasted slots
- Flat arrays REMOVED once per-VC replacements exist (not kept alongside)

### CT args header organization
- Per-VC helpers co-located in whichever file has the flat array they replace (not a new header)
- Foundational per-VC constants (`MAX_NUM_SENDER_CHANNELS_PER_VC`, `vc_sender_channel_start_per_vc`) go in `fabric_erisc_router_ct_args.hpp` next to existing `MAX_NUM_SENDER_CHANNELS_VC0`/`_VC1` definitions (line 82-84)

### Claude's Discretion
- Exact ordering of per-VC helper definitions within each file
- How `std::tuple` type aliases are named for the per-VC runtime state collections
- Whether `vc_sender_channel_start_per_vc` array is still needed after flat arrays are removed (may become unused)
- Implementation details of `for_each_index` constexpr dispatch at each call site

</decisions>

<specifics>
## Specific Ideas

- Reference PR #39538 for target state where decisions above don't override
- `std::tuple` pattern mirrors existing usage in `edm_fabric_flow_control_helpers.hpp` line 270: `std::tuple<ChannelType<BufferSizes[Is]>...> channel_ptrs`
- Function template accessors (`is_sender_channel_serviced_vc<VC, CH>()`) allow the compiler to constant-fold both VC and channel dimensions, enabling dead code elimination per-channel

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `MAX_NUM_SENDER_CHANNELS_VC0`/`_VC1` already derived in `fabric_erisc_router_ct_args.hpp` (line 82-83)
- `VC1_SENDER_CHANNEL_START = MAX_NUM_SENDER_CHANNELS_VC0` already defined (line 84)
- `take_first_n_elements<N, M, T>()` utility in `compile_time_arg_tmp.hpp` — already used to build `sender_ch_live_check_skip` from max-sized array
- `std::tuple` with `std::get<VC>()` pattern in `edm_fabric_flow_control_helpers.hpp`
- `tt::stl::concepts::for_each_index` likely available for constexpr index dispatch

### Established Patterns
- Per-VC arrays use `std::array<T, MAX_NUM_VCS>` throughout host-side code (Phases 1-8)
- `is_sender_channel_serviced[]` flat array used in ~9 `if constexpr` guards in `fabric_erisc_router.cpp`
- `SENDER_NUM_BUFFERS_ARRAY` built via `build_num_slots_array<channel_allocs, ...>()` TMP

### Integration Points
- `any_sender_channels_active` called from `update_telemetry` and perf telemetry block (line 2525)
- `update_telemetry` called from main loop (line 2492)
- Sender init loop (line 1733) is template `<size_t NUM_SENDER_CHANNELS, ...>` — already partially templated
- `execute_main_loop` lambda declares all flat runtime arrays — this is where tuple declarations go
- 8 flat sender channel loops identified at lines: 434, 1387, 2693, 2795, 2945, 2950, 3020, 3109

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-device-side-kernel-per-vc-templates*
*Context gathered: 2026-03-14*
