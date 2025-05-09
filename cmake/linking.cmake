if(TT_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)

    # LTO can be requested but not supported, so handle that case
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO/IPO is supported.")
        set(TT_LTO_ENABLED ON)

        # Enable one-definition-rule (ODR) warnings.
        # Only works when LTO is enabled.
        add_compile_options(-Wodr)
    else()
        message(SEND_ERROR "LTO/IPO is not supported: ${output}")
        set(TT_LTO_ENABLED OFF)
    endif()
else()
    set(TT_LTO_ENABLED OFF)
endif()
