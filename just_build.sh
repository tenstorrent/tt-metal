#!/bin/bash
cmake --preset default -DBUILD_TT_TRAIN=FALSE
cmake --build default --preset dev --clean-first
# this file should be deleted before merging into main
