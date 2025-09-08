##### NOTE: Currently only tested for Ubuntu 22.04

## Build python_env with --site-packages option.
./create_venv_gst.sh

deactivate
./models/experimental/yolo_common/gstreamer_plugins/create_venv_gstreamer.sh
source python_env_gst/bin/activate


sudo apt install python3-gi python3-gi-cairo
sudo apt install python3-gst-1.0 gstreamer1.0-python3-plugin-loader
sudo apt install gstreamer1.0-tools
pip install graphviz numpy_ringbuffer
sudo apt install ubuntu-restricted-extras
sudo apt-get install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-libav

export GST_PLUGIN_PATH=$PWD/models/experimental/yolo_common/gstreamer_plugins/plugin:$PWD/models/experimental/yolo_common/gstreamer_plugins/plugins

rm ~/.cache/gstreamer-1.0/registry.x86_64.bin
gst-inspect-1.0 python

## OUTPUT:
Plugin Details:
  Name                     python
  Description              loader for plugins written in python
  Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstpython.so
  Version                  1.20.1
  License                  LGPL
  Source module            gst-python
  Binary package           GStreamer Python
  Origin URL               http://gstreamer.freedesktop.org


##### Test plugin:

## Yolov9c:
# Segmentation
GST_DEBUG=4 gst-launch-1.0 -v videotestsrc ! "video/x-raw,width=640,height=480,framerate=30/1,format=UYVY" ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=640,format=NV12,framerate=30/1" ! queue ! x264enc tune=zerolatency ! h264parse ! avdec_h264 ! videoconvert ! "video/x-raw,format=BGRx,width=640,height=640,framerate=30/1" ! yolov9c batch-size=1 type=segment ! videoconvert ! x264enc tune=zerolatency ! h264parse ! avdec_h264 ! autovideosink

# Object Detection
GST_DEBUG=4 gst-launch-1.0 -v videotestsrc ! "video/x-raw,width=640,height=480,framerate=30/1,format=UYVY" ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=640,format=NV12,framerate=30/1" ! queue ! x264enc tune=zerolatency ! h264parse ! avdec_h264 ! videoconvert ! "video/x-raw,format=BGRx,width=640,height=640,framerate=30/1" ! yolov9c batch-size=1 type=detection ! videoconvert ! x264enc tune=zerolatency ! h264parse ! avdec_h264 ! autovideosink


## Yolov8x:
GST_DEBUG=4 gst-launch-1.0 -v videotestsrc ! "video/x-raw,width=640,height=480,framerate=30/1,format=UYVY" ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=640,format=NV12,framerate=30/1" ! queue ! x264enc tune=zerolatency ! h264parse ! avdec_h264 ! videoconvert ! "video/x-raw,format=BGRx,width=640,height=640,framerate=30/1" ! yolov8x batch-size=1 ! videoconvert ! x264enc tune=zerolatency ! h264parse ! avdec_h264 ! autovideosink
