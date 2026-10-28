# Compiles case CASE of SOURCE, which must not compile, and passes only when the compiler reports exactly one error,
# MESSAGE, reached from a line of SOURCE.
execute_process(
    COMMAND
        ${COMPILER} ${FLAGS} -fsyntax-only -D${MACRO}=${CASE} ${SOURCE}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
)
string(REGEX MATCHALL "error: " errors "${output}")
list(LENGTH errors error_count)
get_filename_component(source_name ${SOURCE} NAME)
string(FIND "${output}" "${MESSAGE}" message_at)
if(result EQUAL 0 OR NOT error_count EQUAL 1 OR message_at EQUAL -1 OR NOT output MATCHES "${source_name}:[0-9]+:[0-9]+")
    message(FATAL_ERROR "${MACRO} ${CASE}: want one error, \"${MESSAGE}\", got:\n${output}")
endif()
