.. _TT-Metal Models Performance:

Performance
===========

Prerequisites
-------------

Ensure that you have the base TT-Metalium source and environment configuration, and all the requirements for the models are installed. Follow
:ref:`these instructions <TT-Metalium Models Get Started>`.

Running a perf file
-------------------

Each model ready for profiling comes with a perf file, typically found under ``models/YOUR_MODEL/perf_MODEL.py``. To profile a model use

::

    pytest tests/python_api_testing/models/YOUR_MODEL/tests/perf_YOUR_MODEL.py


Perf files will write the results in a csv file ``perf_YOUR_MODEL_date.csv``. This file contains a table with two rows, headers and results


.. list-table::
   :widths: 25 25 25 25 25 25 25 25 25 25
   :header-rows: 1

   * - Model
     - Setting
     - Batch
     - First Run (sec)
     - Second Run (sec)
     - Compile Time (sec)
     - Inference Time GS (sec)
     - Throughput GS (batch*inf/sec)
     - Inference Time CPU (sec)
     - Throughput CPU (batch*inf/sec)
   * - vit
     - base-patch16
     - 1
     - 30.51
     - 16.05
     - 14.46
     - 16.05
     - 0.0623
     - 0.29
     - 3.4960

* **First Run**: Includes compilation time and inference time; without any caching enabled.
* **Second Run** and **Inference Time GS**: Inference time of abovementioned model on Grayskull. It is referred to as Second Run since during the first run we cache the compile program and do not pay for the compilation at the second run.
* **Compile Time**: Compile time as the name suggest, calculated by subtracting Second Run from the First Run.
* **Throughput GS** Throughput of the model on Grayskull, computed as (batch*inf/sec) where inf is inference time on Grayskull.
* **Inference Time CPU** Inference time of abovementioned model on CPU.
* **Throughput cpu** Throughput of the model's implementation of pytorch on CPU, computed as (batch*inf/sec) where inf inference time on CPU.




Running all the perf files
--------------------------

We also maintain ``tests/scripts/run_performance.sh`` to facilitate an easy way to profile all the models. Our attempt is to have the fastest commands for each perf file in this script. You can execute this file

::

    ./tests/scripts/run_performance.sh


This script will run all the perf files and merge the output csv files into on ``perf.csv`` file.
