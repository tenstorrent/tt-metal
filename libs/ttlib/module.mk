include $(TT_METAL_HOME)/libs/ttlib/csrc/module.mk

ttlib: ttlib/csrc ttlib/csrc/setup_local_so
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"
