cd .build/default
rm -rf *.deb
rm -rf *.ddeb

cd ..
cd ..

cmake --preset default -DBUILD_TT_TRAIN=FALSE
cmake --build default --preset dev --clean-first
cd .build/default
ninja package
cd ..
cd ..
docker stop evan-package-testing
docker rm evan-package-testing
docker build -f dockerfile/Dockerfile.basic-dev -t evan-testing .

CONTAINER_NAME="evan-package-testing"
DEB_FILES_DIRECTORY="/home/ebanerjee/tt-metal/.build/default"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker build -f dockerfile/Dockerfile.basic-dev -t evan-testing 

docker run --name "$CONTAINER_NAME" --privileged \
    --device /dev/tenstorrent:/dev/tenstorrent \
    -v /dev/hugepages-1G/:/dev/hugepages-1G/ \
    -v /opt/tenstorrent/sfpi:/opt/tenstorrent/sfpi \
    evan-testing \
