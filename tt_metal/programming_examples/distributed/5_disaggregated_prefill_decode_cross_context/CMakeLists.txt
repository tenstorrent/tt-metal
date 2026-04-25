if(ENABLE_DISTRIBUTED)
    set(EXAMPLE_DISAGGREGATED_PREFILL_DECODE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/example_disaggregated_prefill_decode_cross_context.cpp
    )
    add_executable(example_disaggregated_prefill_decode_cross_context ${EXAMPLE_DISAGGREGATED_PREFILL_DECODE_SRC})
    target_link_libraries(
        example_disaggregated_prefill_decode_cross_context
        PRIVATE
            tt_metal
            OpenMPI::MPI
    )
    target_include_directories(
        example_disaggregated_prefill_decode_cross_context
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
    )
    if(TARGET TT::CommonPCH)
        tt_reuse_precompile_headers(example_disaggregated_prefill_decode_cross_context TT::CommonPCH)
    endif()
    set_target_properties(
        example_disaggregated_prefill_decode_cross_context
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/programming_examples/distributed
    )
endif()
