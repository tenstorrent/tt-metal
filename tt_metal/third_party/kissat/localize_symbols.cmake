# Post-build symbol localization for the vendored kissat static library.
#
# WHY: kissat vendors its own copy of "kitten" (a small sub-solver). CaDiCaL vendors kitten too, and both
# export the same global symbols (kitten_value, kitten_solve, kitten_shrink_to_clausal_core, ... — 20+ of
# them, plus new_learned_klause). While the fabric SAT engine keeps BOTH backends linkable (CaDiCaL is the
# default; kissat is opt-in via TT_TOPO_SAT_ENGINE), both archives land in the same link (libtt_metal.so)
# and the duplicate kitten symbols cause "multiple definition" link errors.
#
# FIX: kissat is only ever called through its public C API (kissat_*). So we demote every OTHER global symbol
# in the archive to local binding, leaving nothing but kissat_* globally visible — the colliding kitten_*
# symbols become file-local and no longer clash with CaDiCaL's copy.
#
# HOW: a static archive's inter-object references are only resolved at final link, so we cannot localize an
# internal-but-cross-referenced global before then without breaking kissat's own linkage. We first partial-
# link (ld -r) all objects into ONE relocatable object — resolving kissat's internal cross-references — and
# only then localize, which is now safe. The archive is repacked to hold just that one localized object.
#
# Invoked as: cmake -DKISSAT_LIB=<path/to/libkissat.a> -DTOOL_AR=<ar> -DTOOL_LD=<ld> -DTOOL_OBJCOPY=<objcopy>
#             -DTOOL_NM=<nm> -P localize_symbols.cmake

foreach(_v KISSAT_LIB TOOL_AR TOOL_LD TOOL_OBJCOPY TOOL_NM)
    if(NOT ${_v})
        message(FATAL_ERROR "localize_symbols.cmake: ${_v} not set")
    endif()
endforeach()

get_filename_component(_lib_abs "${KISSAT_LIB}" ABSOLUTE)
set(_work "${_lib_abs}.localize")
file(REMOVE_RECURSE "${_work}")
file(MAKE_DIRECTORY "${_work}")

# 1) Extract the archive's objects.
execute_process(COMMAND "${TOOL_AR}" x "${_lib_abs}" WORKING_DIRECTORY "${_work}" RESULT_VARIABLE _rc)
if(_rc)
    message(FATAL_ERROR "localize: ar x failed (${_rc})")
endif()
file(GLOB _objs "${_work}/*.o")
if(NOT _objs)
    message(FATAL_ERROR "localize: no objects extracted from ${_lib_abs}")
endif()

# 2) Partial-link into one relocatable object so internal cross-references resolve.
execute_process(COMMAND "${TOOL_LD}" -r -o "${_work}/kissat_combined.o" ${_objs} RESULT_VARIABLE _rc)
if(_rc)
    message(FATAL_ERROR "localize: ld -r failed (${_rc})")
endif()

# 3) Build the keep-list: every defined global whose name starts with kissat_ (the public API).
execute_process(
    COMMAND "${TOOL_NM}" -g --defined-only "${_work}/kissat_combined.o"
    OUTPUT_VARIABLE _nm_out
    RESULT_VARIABLE _rc
)
if(_rc)
    message(FATAL_ERROR "localize: nm failed (${_rc})")
endif()
set(_keep "")
string(REPLACE "\n" ";" _nm_lines "${_nm_out}")
foreach(_line IN LISTS _nm_lines)
    # nm line: "<addr> <type> <name>"; keep public kissat_* symbols global.
    if(_line MATCHES "[0-9a-fA-F]+ [TDBRW] (kissat_[A-Za-z0-9_]+)$")
        list(APPEND _keep "${CMAKE_MATCH_1}")
    endif()
endforeach()
if(NOT _keep)
    message(FATAL_ERROR "localize: found no kissat_* public symbols to keep")
endif()
list(REMOVE_DUPLICATES _keep)
list(LENGTH _keep _keep_len)
string(REPLACE ";" "\n" _keep_text "${_keep}")
file(WRITE "${_work}/keep.syms" "${_keep_text}\n")

# 4) Localize everything except the keep-list.
execute_process(
    COMMAND "${TOOL_OBJCOPY}" "--keep-global-symbols=${_work}/keep.syms"
            "${_work}/kissat_combined.o" "${_work}/kissat_local.o"
    RESULT_VARIABLE _rc
)
if(_rc)
    message(FATAL_ERROR "localize: objcopy failed (${_rc})")
endif()

# 5) Repack the archive with just the single localized object.
file(REMOVE "${_lib_abs}")
execute_process(COMMAND "${TOOL_AR}" rcs "${_lib_abs}" "${_work}/kissat_local.o" RESULT_VARIABLE _rc)
if(_rc)
    message(FATAL_ERROR "localize: ar rcs failed (${_rc})")
endif()

message(STATUS "kissat: localized non-public symbols (kept ${_keep_len} kissat_* globals); kitten_* no longer collide with CaDiCaL")
