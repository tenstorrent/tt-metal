
if [ -n "$CI_PROJECT_DIR" ]; then
  if [ -z $ROOT ]; then export ROOT=$CI_PROJECT_DIR; fi
else
  if [ -z $ROOT ]; then export ROOT=`git rev-parse --show-toplevel`; fi
fi


cd $ROOT/src/ckernels/wormhole/llk/src && make clean
cd $ROOT/src/ckernels/wormhole/llk/src/blank && make clean
