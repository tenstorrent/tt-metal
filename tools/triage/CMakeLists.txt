# Create .capnp directory
set(TRIAGE_CAPNP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.capnp)
file(MAKE_DIRECTORY ${TRIAGE_CAPNP_DIR})
file(MAKE_DIRECTORY ${TRIAGE_CAPNP_DIR}/tt-metalium/experimental)

# List of input capnp files from original locations in repository
set(TRIAGE_INPUT_INSPECTOR_RPC_CAPNP ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium/experimental/inspector_rpc.capnp)
set(TRIAGE_INPUT_RUNTIME_INSPECTOR_RPC_CAPNP ${PROJECT_SOURCE_DIR}/tt_metal/impl/debug/inspector/runtime_rpc.capnp)
set(TRIAGE_INPUT_TTNN_INSPECTOR_RPC_CAPNP ${PROJECT_SOURCE_DIR}/ttnn/core/inspector/ttnn_rpc.capnp)

# List of copied capnp files in this directory
set(TRIAGE_INSPECTOR_RPC_CAPNP ${TRIAGE_CAPNP_DIR}/tt-metalium/experimental/inspector_rpc.capnp)
set(TRIAGE_RUNTIME_INSPECTOR_RPC_CAPNP ${TRIAGE_CAPNP_DIR}/runtime_rpc.capnp)
set(TRIAGE_TTNN_INSPECTOR_RPC_CAPNP ${TRIAGE_CAPNP_DIR}/ttnn_rpc.capnp)

# Copy capnp files to .capnp directory
foreach(
    CAPNP_PAIR
    "${TRIAGE_INPUT_INSPECTOR_RPC_CAPNP};${TRIAGE_INSPECTOR_RPC_CAPNP}"
    "${TRIAGE_INPUT_RUNTIME_INSPECTOR_RPC_CAPNP};${TRIAGE_RUNTIME_INSPECTOR_RPC_CAPNP}"
    "${TRIAGE_INPUT_TTNN_INSPECTOR_RPC_CAPNP};${TRIAGE_TTNN_INSPECTOR_RPC_CAPNP}"
)
    list(GET CAPNP_PAIR 0 INPUT_FILE)
    list(GET CAPNP_PAIR 1 OUTPUT_FILE)
    add_custom_command(
        OUTPUT
            ${OUTPUT_FILE}
        COMMAND
            ${CMAKE_COMMAND} -E copy ${INPUT_FILE} ${OUTPUT_FILE}
        DEPENDS
            ${INPUT_FILE}
        COMMENT "Copying ${INPUT_FILE} to ${OUTPUT_FILE}"
    )
endforeach()

# Generate Inspector RPC stub file
set(TRIAGE_INSPECTOR_RPC_STUB_FILE ${CMAKE_CURRENT_SOURCE_DIR}/inspector_capnp.pyi)
set(TRIAGE_GENERATE_RPC_STUB ${PROJECT_SOURCE_DIR}/scripts/generate_rpc_stub.py)
add_custom_command(
    OUTPUT
        ${TRIAGE_INSPECTOR_RPC_STUB_FILE}
    COMMAND
        python3 ${TRIAGE_GENERATE_RPC_STUB} ${TRIAGE_INSPECTOR_RPC_STUB_FILE} ${TRIAGE_CAPNP_DIR}
        ${TRIAGE_RUNTIME_INSPECTOR_RPC_CAPNP} ${TRIAGE_INSPECTOR_RPC_CAPNP} ${TRIAGE_TTNN_INSPECTOR_RPC_CAPNP}
    DEPENDS
        ${TRIAGE_INSPECTOR_RPC_CAPNP}
        ${TRIAGE_RUNTIME_INSPECTOR_RPC_CAPNP}
        ${TRIAGE_TTNN_INSPECTOR_RPC_CAPNP}
        ${TRIAGE_GENERATE_RPC_STUB}
    COMMENT
        "Generating RPC stub for ${TRIAGE_INSPECTOR_RPC_CAPNP}, ${TRIAGE_RUNTIME_INSPECTOR_RPC_CAPNP}, ${TRIAGE_TTNN_INSPECTOR_RPC_CAPNP}"
)
add_custom_target(
    inspector_rpc_stub
    ALL
    DEPENDS
        ${TRIAGE_INSPECTOR_RPC_STUB_FILE}
        ${TRIAGE_RUNTIME_INSPECTOR_RPC_CAPNP}
        ${TRIAGE_TTNN_INSPECTOR_RPC_CAPNP}
)
