import time
import io
import json
import argparse
import cv2
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        ...

    def cv2_plot_results(self, bgr_image, selected_classes, prob, boxes):
        for selected_class, p, [xmin, ymin, xmax, ymax] in zip(selected_classes, prob, boxes):
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            cv2.rectangle(bgr_image, (xmin, ymin), (xmax, ymax), (45, 200, 200), 2)
            p = int(p * 100)
            text = f"{selected_class}:{p}%"
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1
            lineType = 2
            text_color_bg = (127, 50, 127)
            text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
            text_w, text_h = text_size[0], text_size[1]
            cv2.rectangle(
                bgr_image,
                (xmin - 2, ymin - 2),
                (xmin + text_w + 2, ymin + text_h + 2),
                text_color_bg,
                -1,
            )
            cv2.putText(
                bgr_image,
                text,
                (xmin, ymin + text_h),
                font,
                fontScale,
                fontColor,
                thickness,
            )
        return bgr_image

    def transform(self, frame):
        t0 = time.time()
        pil_image = frame.to_image()
        t1 = time.time()
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        file = {"file": byte_im}
        # Argument Parser to grab namespace_id of server pod from user
        parser = argparse.ArgumentParser(description="YOLOv4 script")
        parser.add_argument("--api-url", type=str, help="URL for the object detection API", required=True)
        args = parser.parse_args()
        apiurl = args.api_url
        url = f"{apiurl}/objdetection_v2"
        r = requests.post(url, files=file)
        data = json.loads(r.content).replace("\n", " ").replace("  ", "")
        data = json.loads(data)
        selected_classes, selected_scores, selected_boxes = (
            data["labels"],
            data["scores"],
            data["bboxes"],
        )
        t3 = time.time()
        bgr_image = frame.to_ndarray(format="bgr24")
        image_final = self.cv2_plot_results(bgr_image, selected_classes, selected_scores, selected_boxes)
        t4 = time.time()
        print()
        print(f" IMG-IN | WH | Post | Total time: ")
        print(f" {(t1-t0):.3f} | {(t3-t1):.3f} | {(t4-t3):.3f} || {(t4-t0):.3f} ")

        return image_final


st.sidebar.image("TT.png", use_column_width=True)
st.sidebar.image("GS.png", use_column_width=True)

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"min": 640, "ideal": 800, "max": 1920},
            "height": {"min": 360, "ideal": 450, "max": 900},
            "frameRate": {"min": 1, "ideal": 20, "max": 40},
        }
    },
)
