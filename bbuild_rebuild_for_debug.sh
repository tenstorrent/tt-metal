#!/bin/bash

cmake --preset default -DBUILD_TT_TRAIN=FALSE
#cmake --build default --preset dev --clean-first -- -k 0
cmake --build default --preset dev --clean-first -- -k 0 > build.log 2>&1
