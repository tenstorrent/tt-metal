
set(ENV{CPM_SOURCE_CACHE} "${CMAKE_SOURCE_DIR}/.cpmcache")

include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)
set(BoostPackages
    Align
    Config
    Container_Hash
    Core
    Detail
    Format
    Interprocess
    Smart_Ptr
    Assert
    Integer
    Type_Traits
    Optional
    Static_Assert
    Throw_Exception
    Move
    Utility
    Preprocessor
    Date_Time
    Numeric_Conversion
    Mpl
)

foreach(package ${BoostPackages})
    CPMAddPackage(
        NAME Boost${package}
        GITHUB_REPOSITORY boostorg/${package}
        GIT_TAG boost-1.76.0
        DOWNLOAD_ONLY YES
    )
endforeach()
