.PHONY: src/ckernels
src/ckernels:
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/ckernels/gen

src/ckernels/clean:
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/ckernels/gen clean
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/ckernels/gen clean
