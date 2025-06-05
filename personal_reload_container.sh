cd .build/default
rm -rf *.deb
rm -rf *.ddeb

cd ..
cd ..

cmake --preset default
cmake --build default --preset dev --clean-first
cd .build/default
ninja package
cd ..
cd ..
docker stop evan-package-testing
docker rm evan-package-testing
docker build -f dockerfile/Dockerfile.basic-dev -t evan-testing .
docker run -it --name evan-package-testing --privileged --device /dev/tenstorrent:/dev/tenstorrent -v /dev/hugepages-1G/:/dev/hugepages-1G/ -v /opt/tenstorrent/sfpi:/opt/tenstorrent/sfpi evan-testing bash -c "export CC=gcc-12 && export CXX=g++-12 && export TT_METAL_HOME=/usr/libexec/tt-metalium &&add-apt-repository -y ppa:mhier/libboost-latest && cd /debians && apt install -y ./*.deb && cd .. && ./personal_cmake_stuff.sh && cd /usr/share/tt-metalium/examples && exec /bin/bash"