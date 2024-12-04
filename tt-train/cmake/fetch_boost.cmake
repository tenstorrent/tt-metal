include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

function(fetch_boost_library BOOST_PROJECT_NAME)
    CPMAddPackage(
        NAME boost_${BOOST_PROJECT_NAME}
        GITHUB_REPOSITORY boostorg/${BOOST_PROJECT_NAME}
        GIT_TAG boost-1.85.0
        OPTIONS
            "BUILD_SHARED_LIBS OFF"
    )

    get_target_property(BOOST_INTERFACE_LINK_LIBRARIES boost_${BOOST_PROJECT_NAME} INTERFACE_LINK_LIBRARIES)

    if(NOT BOOST_INTERFACE_LINK_LIBRARIES STREQUAL BOOST_INTERFACE_LINK_LIBRARIES-NOTFOUND)
        foreach(BOOST_INTERFACE_LINK_LIBRARY IN ITEMS ${BOOST_INTERFACE_LINK_LIBRARIES})
            if(
                NOT TARGET
                    ${BOOST_INTERFACE_LINK_LIBRARY}
                AND BOOST_INTERFACE_LINK_LIBRARY
                    MATCHES
                    "^Boost::([a-z0-9_]+)$"
            )
                fetch_boost_library(${CMAKE_MATCH_1})
            endif()
        endforeach()
    endif()
endfunction()
