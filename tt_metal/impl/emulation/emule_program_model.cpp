// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "emule_program_model.hpp"

#include "emule_jit.hpp"
#include "emule_kernel_defines.hpp"
#include "emule_metal2_emit.hpp"

#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/hal.hpp>
#include "jit_build/jit_build_settings.hpp"
#include "jit_build/jit_build_utils.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::emule {

void collect_kernels(
    detail::ProgramImpl& impl,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base,
    const std::string& extra_inc,
    std::map<CoreCoord, std::vector<PendingKernelInfo>>& pending_core_kernels,
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps) {
    static const char* trisc_define_names[] = {"TRISC_UNPACK", "TRISC_MATH", "TRISC_PACK", "TRISC_ISOLATE_SFPU"};

    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_pct = hal.get_programmable_core_type_count();
    for (uint32_t pct = 0; pct < num_pct; ++pct) {
        auto& kernels = impl.get_kernels(pct);
        // (kernel_id, logical_core) → kg: a kernel that runs on cores in
        // multiple KGs has distinct per-KG launch_msg layouts, so keying by
        // kernel alone picks the wrong one for half the cores.
        std::map<std::pair<KernelHandle, CoreCoord>, KernelGroup*> kernel_core_to_kg;
        for (const auto& kg : impl.get_kernel_groups(pct)) {
            for (const auto& cr : kg->core_ranges.ranges()) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
                    for (auto y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                        CoreCoord lc(x, y);
                        for (auto kid : kg->kernel_ids) {
                            kernel_core_to_kg.emplace(std::make_pair(kid, lc), kg.get());
                        }
                    }
                }
            }
        }
        for (auto& [kernel_id, kernel] : kernels) {
            const auto& ksrc = kernel->kernel_source();
            std::string src_path = resolve_kernel_source_path(ksrc, inline_src_temps);
            if (ksrc.source_type_ == KernelSource::FILE_PATH) {
                src_path = resolve_emule_kernel_source_shadow(src_path, impl.get_context_id());
            }

            // Thread each kernel's configured include roots into its JIT -I flags.
            // Kernels can declare extra include paths (Kernel::process_include_paths)
            // so root-rooted includes resolve at compile time; silicon's build wires
            // these through the compiler include dirs, so mirror that here.
            std::string kernel_extra_inc = extra_inc;
            kernel->process_include_paths(
                [&kernel_extra_inc](const std::string& p) { kernel_extra_inc += " -I\"" + p + "\""; });

            auto compile_args = kernel->compile_time_args();
            auto named_compile_args = kernel->named_compile_time_args();
            ////////////////////////////////////////////////////////////
            // Blaze-only experimental named args
            // Removal is tracked by issue #50953
            NamedCTArgNamespaces named_ct_arg_namespaces;
            kernel->process_named_ct_arg_namespaces([&named_ct_arg_namespaces](const NamedCTArgNamespaces& namespaces) {
                named_ct_arg_namespaces = namespaces;
            });
            NamedRuntimeArgNamespaces named_runtime_arg_namespaces;
            kernel->process_named_runtime_args(
                [&named_runtime_arg_namespaces](const NamedRuntimeArgNamespaces& namespaces) {
                    named_runtime_arg_namespaces = namespaces;
                });
            ////////////////////////////////////////////////////////////
            auto defines = build_kernel_defines(
                *kernel, impl, num_dram_channels, num_l1_banks, worker_col_map_str, worker_row_map_str, emule_sem_base);

            // Tensix/compute kernels use bits 8+ in the DFB RISC mask (TENSIX_RISC_OFFSET),
            // while DM kernels use bits 0-7 directly.
            bool is_tensix = (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE);
            auto* qdm = dynamic_cast<experimental::quasar::QuasarDataMovementKernel*>(kernel.get());
            auto* qck = dynamic_cast<experimental::quasar::QuasarComputeKernel*>(kernel.get());
            bool is_quasar_compute = is_tensix && (qck != nullptr);

            // Issue tenstorrent/tt-emule#24: emule runs all three TRISC code paths
            // in a single unified compute thread (no separate UNPACK/MATH/PACK
            // RISC cores). tt-mlir-generated D2M kernels guard hardware code
            // paths with `#ifdef TRISC_UNPACK|MATH|PACK`; without these defines
            // helpers like `experimental::write_row_mask_tile` compile to empty
            // bodies and the downstream `where_tile` reads stale data. Define
            // all three so the kernel's `#ifdef TRISC_*` blocks execute exactly
            // once on the unified thread.
            if (is_tensix && !is_quasar_compute) {
                defines["TRISC_UNPACK"] = "1";
                defines["TRISC_MATH"] = "1";
                defines["TRISC_PACK"] = "1";
            }

            // Metal 2.0 bindings — same across this Kernel's TRISC variants, so
            // capture the cache-key suffix once and append it to every variant key.
            Metal2BindingsSnapshot bindings = build_metal2_snapshot(*kernel);
            for (const auto& [sem_name, h] : bindings.sem_accessors) {
                TT_FATAL(
                    h.scope != SemScope::DM_LOCAL_CACHED,
                    "Internal error: semaphore '{}' resolved to DM_LOCAL_CACHED under emule, but the emule "
                    "backend does not model the cached pool (no seeder is emitted); the classifier "
                    "(ResolveSemaphoreScope) must never pick the cached tier for this backend.",
                    sem_name);
            }
            const std::string metal2_key_suffix = bindings.cache_key_suffix();

            // GENERAL emule fix — intentionally NOT part of the Blaze named-args feature and NOT
            // fenced: it is required by any compute or data-movement kernel built under emule and
            // must remain in place after the named-args feature is deleted (issue #50953). It is
            // grouped here only because it shares this per-kernel `defines` map.
            //
            // COMPILE_FOR_{TRISC,BRISC,NCRISC} defines — silicon's per-RISC kernel build sets
            // exactly one of these. Kernel-author API headers use them to pick the right include
            // chain and to define `is_brisc` / `is_ncrisc` / `is_trisc` constexpr bools; without
            // them, `SelectByRISCV<>` aliases fail to resolve. Emule runs all RISCs in one unified
            // thread, so we set the corresponding macro based on the kernel's processor class.
            //
            // PROCESSOR_INDEX backs get_hw_thread_idx(), so the debug headers that reach it
            // (waypoint, pause, assert, device_print) will not compile without it.
            uint32_t processor_type_idx = 0;  // emule fuses TRISC0-2 into one compute fiber
            if (is_tensix) {
                defines["COMPILE_FOR_TRISC"] = "1";
            } else if (auto* dm_kernel = dynamic_cast<DataMovementKernel*>(kernel.get()); dm_kernel != nullptr) {
                auto cfg_variant = dm_kernel->config();
                const auto& cfg = std::get<DataMovementConfig>(cfg_variant);
                switch (cfg.processor) {
                    case DataMovementProcessor::RISCV_0:
                        defines["COMPILE_FOR_BRISC"] = "1";
                        processor_type_idx = 0;
                        break;
                    case DataMovementProcessor::RISCV_1:
                        defines["COMPILE_FOR_NCRISC"] = "1";
                        processor_type_idx = 1;
                        break;
                    default: break;
                }
            }
            defines["PROCESSOR_INDEX"] = std::to_string(hal.get_processor_index(
                hal.get_programmable_core_type(pct), kernel->get_kernel_processor_class(), processor_type_idx));

            // Helper: compute cache key from a defines map (preserves upstream's sorted
            // iteration of named_compile_args and defines for key stability).
            auto compute_cache_key = [&](const std::map<std::string, std::string>& defs) -> std::string {
                std::string key;
                if (ksrc.source_type_ == KernelSource::FILE_PATH) {
                    key = src_path;
                } else {
                    char hex[FNV_HEX_BUF_SIZE];
                    std::snprintf(hex, sizeof(hex), "%016lx", fnv1a_hash(ksrc.source_));
                    key = std::string("inline:") + hex;
                }
                for (auto v : compile_args) {
                    key += ":" + std::to_string(v);
                }
                std::vector<std::pair<std::string, uint32_t>> sorted_named(
                    named_compile_args.begin(), named_compile_args.end());
                std::sort(sorted_named.begin(), sorted_named.end());
                for (const auto& [k, v] : sorted_named) {
                    key += ":N" + k + "=" + std::to_string(v);
                }
                ////////////////////////////////////////////////////////////
                // Blaze-only experimental named args
                // Removal is tracked by issue #50953
                // The compiled wrapper depends on named_runtime_arg_namespaces through
                // named_args_generated.h (the blaze_rt_args:: Arg/ArrayArg descriptors),
                // so the full named RT schema — ns, field, index, length, dispatch — must
                // be part of the key. Without it, two kernels sharing source/CT args/defines
                // but differing in Blaze RT names or layout alias in the JIT and disk caches
                // and load a stale descriptor layout (the .so then reads runtime args from
                // the wrong slots). Typed CT args bypass named_compile_args, so serialize them too.
                // Both namespace maps have a fixed iteration order: namespaces are sorted
                // and entries retain declaration order. Names cannot contain the ':', '=',
                // or ',' separators used below.
                for (const auto& [ns, entries] : named_ct_arg_namespaces) {
                    key += ":bctns:" + ns;
                    for (const auto& [field, value] : entries) {
                        key += ":bct:" + field + "=" + std::to_string(value);
                    }
                }
                for (const auto& [ns, entries] : named_runtime_arg_namespaces) {
                    key += ":brtns:" + ns;
                    for (const auto& entry : entries) {
                        key += ":brt:" + entry.field + "=" + std::to_string(entry.index) + "," +
                               std::to_string(entry.length) + "," +
                               std::to_string(static_cast<uint32_t>(entry.dispatch));
                    }
                }
                ////////////////////////////////////////////////////////////
                for (const auto& [k, v] : defs) {
                    key += ":" + k + "=" + v;
                }
                key += metal2_key_suffix;
                // Per-kernel include roots (Kernel::process_include_paths) change
                // which headers resolve, and therefore the compiled artifact, so
                // fold them into the key. Without this, two kernels that share
                // src_path/compile_args/named_compile_args/defines but differ in
                // include configuration would alias in the JIT cache (in-memory
                // g_jit_cache and deferred_compiles) and a disk-cached .so built
                // under a different include config could be reused. Kernels with no
                // extra include roots keep their previous key (backward-compatible).
                if (kernel_extra_inc != extra_inc) {
                    char inc_hex[FNV_HEX_BUF_SIZE];
                    std::snprintf(inc_hex, sizeof(inc_hex), "%016lx", fnv1a_hash(kernel_extra_inc));
                    key += ":inc";
                    key += inc_hex;
                }
                // ASAN builds are -g; keep their cache distinct from the lean build.
                if (emule_asan_enabled()) {
                    key += ":asan_g";
                }
                return key;
            };

            auto register_cache_key = [&](const std::string& key, const std::map<std::string, std::string>& defs) {
                std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
                auto it = g_jit_cache.find(key);
                if (it != g_jit_cache.end()) {
                    resolved_fns[key] = it->second;
                } else if (
                    resolved_fns.find(key) == resolved_fns.end() &&
                    deferred_compiles.find(key) == deferred_compiles.end()) {
                    std::string mtime_path = (ksrc.source_type_ == KernelSource::FILE_PATH) ? src_path : "";
                    auto disk_fn = disk_cache_lookup(key, mtime_path);
                    if (disk_fn) {
                        resolved_fns[key] = disk_fn;
                        g_jit_cache[key] = disk_fn;
                    } else {
                        deferred_compiles[key] = DeferredCompile{
                            src_path,
                            compile_args,
                            named_compile_args,
                            // Blaze-only experimental named args (issue #50953) — begin
                            named_ct_arg_namespaces,
                            named_runtime_arg_namespaces,
                            // Blaze-only experimental named args (issue #50953) — end
                            defs,
                            kernel_extra_inc,
                            bindings};
                    }
                }
            };

            // For Quasar compute kernels with TRISC guards, compile 4 variants
            // (UNPACK, MATH, PACK, ISOLATE_SFPU) — each has a different cache key.
            // Kernels without TRISC guards (e.g. DFB compute bridges) compile once.
            TriscMode trisc = detect_quasar_trisc_mode(is_quasar_compute, src_path);
            std::vector<std::string> variant_cache_keys;
            bool run_all_variants = false;
            if (trisc.needs_trisc_compile) {
                for (int t = 0; t < 4; t++) {
                    auto trisc_defs = defines;
                    trisc_defs[trisc_define_names[t]] = "1";
                    std::string key = compute_cache_key(trisc_defs);
                    register_cache_key(key, trisc_defs);
                    variant_cache_keys.push_back(std::move(key));
                }
                run_all_variants = true;
            } else {
                std::string key = compute_cache_key(defines);
                register_cache_key(key, defines);
                if (trisc.needs_runtime_trisc) {
                    variant_cache_keys.assign(4, key);
                    run_all_variants = true;
                } else {
                    variant_cache_keys.push_back(std::move(key));
                }
            }

            ProcIdList procs = compute_proc_ids_and_thread_count(*kernel, qdm, qck);

            const uint32_t processor_index = hal.get_processor_index(
                kernel->get_kernel_programmable_core_type(),
                kernel->get_kernel_processor_class(),
                kernel->get_kernel_processor_type(0));

            const auto& core_range_set = kernel->core_range_set();
            for (const auto& core_range : core_range_set.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                        CoreCoord logical_core(x, y);
                        // KG lookup is per (kernel, logical_core): same kernel
                        // on different cores may sit in different KGs with
                        // distinct kernel_config layouts.
                        auto kg_it = kernel_core_to_kg.find({kernel_id, logical_core});
                        uint32_t kernel_config_base = 0;
                        uint16_t rta_off = kRtaCrtaNoArgsSentinel;
                        uint16_t crta_off = kRtaCrtaNoArgsSentinel;
                        if (kg_it != kernel_core_to_kg.end()) {
                            auto kc = kg_it->second->launch_msg.view().kernel_config();
                            kernel_config_base = static_cast<uint32_t>(kc.kernel_config_base()[pct]);
                            auto rta = kc.rta_offset()[processor_index];
                            rta_off = rta.rta_offset();
                            crta_off = rta.crta_offset();
                        }
                        // Runtime-arg values (unique + common) for this kernel on this
                        // core; the Object-Intent check uses them to find its I/O tensors
                        // (see ObjectIntentTracker::pre_launch_snapshot). Build once, copy.
                        std::vector<uint32_t> rt_arg_values;
                        uint32_t num_unique_rt = 0;
                        if (kernel->cores_with_runtime_args().count(logical_core) != 0) {
                            const auto& ra = kernel->runtime_args(logical_core);
                            num_unique_rt = static_cast<uint32_t>(ra.size());
                            rt_arg_values.insert(rt_arg_values.end(), ra.begin(), ra.end());
                        }
                        const auto& cra = kernel->common_runtime_args();
                        rt_arg_values.insert(rt_arg_values.end(), cra.begin(), cra.end());

                        uint8_t tidx = 0;
                        for (uint8_t proc_id : procs.proc_ids) {
                            pending_core_kernels[logical_core].push_back(PendingKernelInfo{
                                variant_cache_keys,
                                run_all_variants,
                                proc_id,
                                tidx++,
                                is_tensix,
                                procs.num_threads,
                                kernel_config_base,
                                rta_off,
                                crta_off,
                                rt_arg_values,
                                src_path,
                                num_unique_rt});
                        }
                    }
                }
            }
        }
    }
}

}  // namespace tt::tt_metal::emule
