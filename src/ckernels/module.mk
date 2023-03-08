.PHONY: src/ckernels
src/ckernels:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/ckernels/gen

src/ckernels/clean:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/ckernels/gen clean
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/ckernels/gen clean
