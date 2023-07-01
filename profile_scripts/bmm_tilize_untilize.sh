#!/bin/bash

pytest tests/python_api_testing/unit_testing/test_bmm_tilize_untilize.py

# # print out noc_async_read src/dest addr
# transfer_size=64
# pytest tests/python_api_testing/unit_testing/test_bmm_tilize_untilize.py > bmm_tilize_untilize.log
# python3 tt_metal/tools/profiler/custom_profile.py bmm_tilize_untilize.log > log/bmm_tilize_untilize_$transfer_size.log
