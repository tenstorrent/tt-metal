#!/usr/bin/env bash

set -o nounset
set -x

# Check if image exists and remove it
if docker images -q docker-conan-test > /dev/null 2>&1; then
    docker rmi docker-conan-test
fi

docker build --progress=plain -f dockerfile/Dockerfile.conan-test.ubuntu -t docker-conan-test .
docker run -it --rm -v $(pwd):/workspace docker-conan-test ./build_metal_with_conan.sh /tt-metal-conan/conan-build
