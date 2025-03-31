project(mnist_mlp)

set(SOURCES
    main.cpp
    utils.cpp
    model.cpp
)
CPMAddPackage(NAME mnist_dataset GITHUB_REPOSITORY wichtounet/mnist GIT_TAG master)
include_directories(${mnist_dataset_SOURCE_DIR}/include)

# Add executable and link libraries
add_executable(mnist_mlp ${SOURCES})
target_link_libraries(mnist_mlp PRIVATE ttml)
target_compile_definitions(mnist_mlp PRIVATE MNIST_DATA_LOCATION="${mnist_dataset_SOURCE_DIR}/")
add_definitions(-DCONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs")
