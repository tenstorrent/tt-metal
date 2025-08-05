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

sudo apt install ./.build/default/tt-metalium_*.deb
sudo apt install ./.build/default/tt-nn_*.deb
sudo apt install ./.build/default/tt-nn-validation_*.deb
