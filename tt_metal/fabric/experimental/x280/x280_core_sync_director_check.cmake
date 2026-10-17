# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# cmake -DCOMPILER=<riscv gcc> -DFLAGS=<flags;...> -DWORK_DIR=<dir> -P x280_core_sync_director_check.cmake
#
# A DRISC (MetalTransport<DRAM>) core_sync director is refused on kMayDirect, and that refusal is
# attributable to the core type: a TENSIX director built from the identical config does not draw it,
# and kMayDirect admits every other core family. MetalTransport's hardware arm is RV32-only, so under
# the RV64 X280 compiler no MetalTransport barrier instantiates and only the refusal text is
# compared.

set(refusal "may not coordinate as director")
file(MAKE_DIRECTORY "${WORK_DIR}")

function(_compile name source out_result out_output)
    file(WRITE "${WORK_DIR}/${name}.cpp" "${source}")
    execute_process(
        COMMAND
            "${COMPILER}" ${FLAGS} -fsyntax-only -x c++ "${WORK_DIR}/${name}.cpp"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE output
    )
    set(${out_result} "${result}" PARENT_SCOPE)
    set(${out_output} "${output}" PARENT_SCOPE)
endfunction()

function(_director core out_source)
    set(${out_source}
        "#include \"core_sync_barrier.hpp\"
#include \"core_sync_transport_metal.hpp\"
using Transport = core_sync::MetalTransport<ProgrammableCoreType::${core}>;
struct DirectorConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = true;
    static constexpr uint32_t kWorkerIndex = 0;
    static constexpr uint32_t kTotalWorkers = 1;
    static constexpr uint32_t kDirectorNocX = 0;
    static constexpr uint32_t kDirectorNocY = 0;
    static constexpr uint32_t kDirectorAddr = 0;
    static constexpr uint32_t kLocalReleaseAddr = 0;
    static constexpr uint32_t kTlbBase = 0;
    static constexpr const Transport::Endpoint* kWorkers = nullptr;
    static constexpr Transport::Endpoint kDirectorEndpoint{};
    static constexpr Transport::Roster kRoster{nullptr, 0u};
};
void core_sync_director_probe() {
    (void)core_sync::core_sync_barrier<DirectorConfig, Transport>::sync(0, 0, 0);
}
"
        PARENT_SCOPE
    )
endfunction()

_director(DRAM source)
_compile(dram_director "${source}" result output)
if(result EQUAL 0)
    message(FATAL_ERROR "MetalTransport<DRAM> director compiled when it must be refused")
endif()
string(FIND "${output}" "${refusal}" at)
if(at LESS 0)
    message(FATAL_ERROR "MetalTransport<DRAM> director was refused, but not on kMayDirect:\n${output}")
endif()

_director(TENSIX source)
_compile(tensix_director "${source}" result output)
string(FIND "${output}" "${refusal}" at)
if(at GREATER_EQUAL 0)
    message(
        FATAL_ERROR
        "MetalTransport<TENSIX> director also drew the kMayDirect refusal, so the DRAM refusal is not attributable to the core type:\n${output}"
    )
endif()

_compile(
    may_direct
    "#include \"core_sync_transport_metal.hpp\"
using core_sync::MetalTransport;
static_assert(!MetalTransport<ProgrammableCoreType::DRAM>::kMayDirect, \"a DRISC core must not be admitted as director\");
static_assert(MetalTransport<ProgrammableCoreType::TENSIX>::kMayDirect, \"a Tensix core must be admitted as director\");
static_assert(MetalTransport<ProgrammableCoreType::ACTIVE_ETH>::kMayDirect, \"an active-ethernet core must be admitted as director\");
static_assert(MetalTransport<ProgrammableCoreType::IDLE_ETH>::kMayDirect, \"an idle-ethernet core must be admitted as director\");
"
    result
    output
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "MetalTransport::kMayDirect does not admit every non-DRAM core family:\n${output}")
endif()
message(STATUS "MetalTransport<DRAM> is refused as director on kMayDirect, and the other core families are admitted")
