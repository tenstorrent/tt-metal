# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI U.S. Corp.
# SPDX-License-Identifier: Apache-2.0

"""Small browser UI for SAM2 image, video, and webcam tracking on Wormhole N300."""

import argparse
import json
import mimetypes
import signal
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

import ttnn
from models.demos.vision.segmentation.sam2.common import load_sam2_model_and_processor
from models.demos.vision.segmentation.sam2.tt.tt_sam2_video import SAM2_L1_SMALL_SIZE, build_tt_sam2_model

MODEL_IMAGE_SIZE = 1024
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MAX_UPLOAD_BYTES = 2 * 1024**3
MAX_WEBCAM_FRAME_BYTES = 8 * 1024**2
# Keep these two values aligned with LIVE_MAX_DEPTH and LIVE_BATCH in PAGE.
LIVE_PIPELINE_DEPTH = 6
LIVE_BATCH_SIZE = 5
VIDEO_POST_DEPTH = 6
MAX_POINTS = 16
MASK_COLOR = np.array([36, 196, 255], dtype=np.float32)
MASK_LUT = (0.55 * np.arange(256, dtype=np.float32)[:, None] + 0.45 * MASK_COLOR).astype(np.uint8).reshape(256, 1, 3)
IMAGE_MEAN_255 = (123.675, 116.28, 103.53)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
PAGE = b"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM2 N300</title>
<style>
body{font:16px system-ui;margin:2rem;background:#171717;color:#eee} main{max-width:1000px;margin:auto}
button,input,select{font:inherit;margin:0 .5rem .5rem 0} #status{margin:.5rem 0;color:#bbb}
progress{width:min(500px,90vw)} #stage{position:relative;display:inline-block;margin-top:.5rem}
#stage img,#stage video{display:block;max-width:95vw;max-height:70vh} #prompt{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
a{display:block;margin-top:1rem;color:#65b8ff} [hidden]{display:none!important}
</style>
<main>
<h1>SAM2 N300</h1>
<input id="file" type="file" accept="image/*,video/*">
<button id="webcam">Use webcam</button>
<select id="tool" disabled>
  <option value="positive">Positive point</option>
  <option value="negative">Negative point</option>
  <option value="box">Box</option>
</select>
<button id="clear" disabled>Clear prompts</button>
<button id="track" hidden disabled>Run tracking</button>
<div id="status">Choose an image, video, or webcam.</div>
<progress id="progress" hidden></progress>
<div id="stage" hidden><img id="image"><video id="video" muted playsinline hidden></video><canvas id="prompt"></canvas></div>
<a id="download" hidden>Download result</a>
</main>
<script>
const file=document.querySelector('#file'),webcam=document.querySelector('#webcam'),tool=document.querySelector('#tool'),clear=document.querySelector('#clear');
const track=document.querySelector('#track'),status=document.querySelector('#status'),progress=document.querySelector('#progress');
const stage=document.querySelector('#stage'),image=document.querySelector('#image'),video=document.querySelector('#video');
const prompt=document.querySelector('#prompt'),download=document.querySelector('#download'),capture=document.createElement('canvas');
// LIVE_MAX_DEPTH and LIVE_BATCH mirror the server's bounded reorder window.
const LIVE_MIN_DEPTH=2,LIVE_START_DEPTH=3,LIVE_MAX_DEPTH=6,LIVE_BATCH=5,LIVE_FRAME_MS=1000/30;
const LIVE_IDLE_SLACK_MS=2,LIVE_QUEUE_STREAK=3,LIVE_DEPTH_HOLD_BATCHES=6;
let points=[],boxPrompt=null,dragStart=null,draftBox=null,dragPointer=null,sourceWidth=0,sourceHeight=0;
let loaded=false,videoInput=false,webcamInput=false,maskReady=false,busy=false,finished=false,operation=0;
let live=false,camera=null,liveTask=null,liveImageUrl=null,liveRequests=new Set();
async function api(url,options={}){
  options.headers={...options.headers,'X-SAM2-Request':'1'};
  const response=await fetch(url,options),value=await response.json();
  if(!response.ok)throw Error(value.error||response.statusText);
  return value;
}
function controls(){
  file.disabled=webcam.disabled=busy||live;tool.disabled=busy||live||!loaded||finished;
  clear.disabled=busy||live||!loaded||(!points.length&&!boxPrompt&&!finished);
  track.hidden=!videoInput;track.disabled=busy||(!live&&(!maskReady||finished));track.textContent=live?'Stop tracking':'Run tracking';
  prompt.style.pointerEvents=busy||live||!loaded||finished?'none':'auto';
}
function setBusy(value,message){if(value)operation++;busy=value;if(message)status.textContent=message;if(!value)progress.hidden=true;controls();return operation;}
function configureStage(width,height){
  sourceWidth=width;sourceHeight=height;const scale=Math.min(1,1600/width,1200/height);
  prompt.width=Math.max(1,Math.round(width*scale));prompt.height=Math.max(1,Math.round(height*scale));
}
function resetSelection(){
  points=[];boxPrompt=dragStart=draftBox=dragPointer=null;loaded=maskReady=finished=false;
  download.hidden=true;stage.hidden=true;prompt.hidden=false;
  if(liveImageUrl){URL.revokeObjectURL(liveImageUrl);liveImageUrl=null;}
}
function stopCamera(){
  if(camera)for(const mediaTrack of camera.getTracks())mediaTrack.stop();
  camera=null;video.srcObject=null;webcam.textContent='Use webcam';
}
function cameraFrame(){
  if(!camera||!video.videoWidth)throw Error('Webcam is not ready.');
  if(capture.width!==sourceWidth||capture.height!==sourceHeight){capture.width=sourceWidth;capture.height=sourceHeight;}
  capture.getContext('2d').drawImage(video,0,0,sourceWidth,sourceHeight);
  return new Promise((resolve,reject)=>capture.toBlob(blob=>blob?resolve(blob):reject(Error('Cannot capture webcam frame.')),'image/jpeg',.75));
}
function nextCameraFrame(){return new Promise(resolve=>{let done=false,id;const videoFrame=!!video.requestVideoFrameCallback,finish=()=>{if(!done){done=true;clearTimeout(timeout);resolve();}},timeout=setTimeout(()=>{videoFrame?video.cancelVideoFrameCallback?.(id):cancelAnimationFrame(id);finish();},100);id=videoFrame?video.requestVideoFrameCallback(finish):requestAnimationFrame(finish);});}
function coordinates(event){
  const area=prompt.getBoundingClientRect();
  return [Math.max(0,Math.min(sourceWidth-1,Math.round((event.clientX-area.left)*sourceWidth/area.width))),
          Math.max(0,Math.min(sourceHeight-1,Math.round((event.clientY-area.top)*sourceHeight/area.height)))];
}
function orderedBox(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
function draw(){
  const context=prompt.getContext('2d');context.clearRect(0,0,prompt.width,prompt.height);if(!sourceWidth)return;
  const radius=Math.max(6,Math.min(sourceWidth,sourceHeight)/80);context.save();context.scale(prompt.width/sourceWidth,prompt.height/sourceHeight);
  for(const [x,y,label] of points){
    context.beginPath();context.arc(x,y,radius,0,2*Math.PI);context.fillStyle=label?'#00ff66':'#ff4040';context.fill();
    context.lineWidth=Math.max(2,radius/3);context.strokeStyle='white';context.stroke();
  }
  const box=draftBox||boxPrompt;
  if(box){context.lineWidth=Math.max(3,radius/3);context.strokeStyle='#ffd84d';context.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
  context.restore();
}
async function showImage(source){image.src=source;await image.decode();image.hidden=false;video.hidden=true;stage.hidden=false;draw();}
async function segment(){
  const previousMask=maskReady;let success=false;maskReady=false;download.hidden=true;setBusy(true,'Sending prompts...');
  try{
    const result=await api('/segment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points,box:boxPrompt})});
    await showImage('/mask?v='+Date.now());maskReady=success=true;
    if(!videoInput){download.href='/mask?v='+Date.now();download.download=result.filename;download.hidden=false;}
    status.textContent=videoInput?'Mask ready. Add prompts to refine it, or run tracking.':'Mask ready. Add prompts to refine it.';
  }catch(error){maskReady=previousMask;status.textContent=error.message;}
  finally{setBusy(false);}
  return success;
}
async function addPoint(event,label){
  if(busy||!loaded||finished)return;
  if(points.length>=16){status.textContent='At most 16 points are supported.';return;}
  const [x,y]=coordinates(event);points.push([x,y,label]);draw();if(!await segment()){points.pop();draw();}
}
prompt.onclick=event=>{if(tool.value==='positive')addPoint(event,1);else if(tool.value==='negative')addPoint(event,0);};
prompt.oncontextmenu=event=>{event.preventDefault();addPoint(event,0);};
function cancelDrag(){dragStart=draftBox=dragPointer=null;draw();}
prompt.onpointerdown=event=>{
  if(busy||tool.value!=='box')return;
  dragPointer=event.pointerId;dragStart=coordinates(event);draftBox=[...dragStart,...dragStart];prompt.setPointerCapture(event.pointerId);draw();
};
prompt.onpointermove=event=>{if(dragStart&&event.pointerId===dragPointer){draftBox=orderedBox(dragStart,coordinates(event));draw();}};
prompt.onpointerup=async event=>{
  if(!dragStart||event.pointerId!==dragPointer)return;
  const candidate=orderedBox(dragStart,coordinates(event)),previousBox=boxPrompt;dragStart=draftBox=dragPointer=null;
  if(candidate[2]-candidate[0]>2&&candidate[3]-candidate[1]>2){boxPrompt=candidate;draw();if(!await segment()){boxPrompt=previousBox;draw();}}else draw();
};
prompt.onpointercancel=cancelDrag;prompt.onlostpointercapture=()=>{if(dragStart)cancelDrag();};
webcam.onclick=async()=>{
  if(camera&&webcamInput){
    resetSelection();videoInput=webcamInput=false;image.hidden=true;video.hidden=false;stage.hidden=false;prompt.hidden=true;
    webcam.textContent='Capture frame';status.textContent='Position the camera, then capture a frame.';controls();return;
  }
  if(!camera){
    resetSelection();videoInput=webcamInput=false;video.pause();video.removeAttribute('src');video.load();setBusy(true,'Requesting webcam access...');
    try{
      if(!navigator.mediaDevices?.getUserMedia)throw Error('Webcam access requires a browser on localhost or HTTPS.');
      camera=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1024,max:1024},height:{ideal:576,max:576}},audio:false});
      video.srcObject=camera;video.controls=false;video.hidden=false;image.hidden=true;stage.hidden=false;prompt.hidden=true;await video.play();
      configureStage(video.videoWidth,video.videoHeight);webcam.textContent='Capture frame';status.textContent='Position the camera, then capture a frame.';
    }catch(error){stopCamera();status.textContent=error.message;}
    finally{setBusy(false);}
    return;
  }
  setBusy(true,'Capturing webcam frame...');
  try{
    const frame=await cameraFrame();
    const info=await api('/upload?name=webcam.jpg&webcam=1',{method:'POST',body:frame});
    resetSelection();videoInput=info.video;webcamInput=info.webcam;configureStage(info.width,info.height);prompt.hidden=false;
    await showImage('/preview?v='+Date.now());loaded=true;webcam.textContent='Retake';
    status.textContent='Add a positive point or draw a box. Add negative points to refine it.';
  }catch(error){stage.hidden=false;video.hidden=false;prompt.hidden=true;status.textContent=error.message;}
  finally{setBusy(false);}
};
file.onchange=async()=>{
  const selected=file.files[0];if(!selected)return;file.value='';
  stopCamera();resetSelection();videoInput=webcamInput=false;video.pause();video.removeAttribute('src');video.load();setBusy(true,'Uploading...');
  try{
    const info=await api('/upload?name='+encodeURIComponent(selected.name),{method:'POST',body:selected});
    videoInput=info.video;configureStage(info.width,info.height);
    await showImage('/preview?v='+Date.now());loaded=true;
    status.textContent='Add a positive point or draw a box. Add negative points to remove unwanted regions.';
  }catch(error){status.textContent=error.message;}
  finally{setBusy(false);}
};
clear.onclick=async()=>{
  setBusy(true,'Clearing prompts...');
  try{
    await api('/clear',{method:'POST'});points=[];boxPrompt=null;maskReady=finished=false;download.hidden=true;prompt.hidden=false;
    if(!webcamInput){video.pause();video.removeAttribute('src');video.load();}
    await showImage('/preview?v='+Date.now());status.textContent='Prompts cleared. Select the object again.';
  }catch(error){status.textContent=error.message;}
  finally{setBusy(false);}
};
function closeWebcamUi(message){
  stopCamera();loaded=maskReady=videoInput=webcamInput=false;finished=true;prompt.hidden=true;status.textContent=message;controls();
}
async function stopWebcamServer(){
  try{await api('/webcam/stop',{method:'POST'});return null;}catch(error){return error;}
}
async function finishWebcam(message,stopping){
  const error=await stopping;closeWebcamUi(error?message+' '+error.message:message);
}
function packLiveFrames(frames){
  const header=new ArrayBuffer(4*frames.length),view=new DataView(header);frames.forEach((frame,index)=>view.setUint32(4*index,frame.size));
  return new Blob([header,...frames]);
}
function unpackLiveFrames(buffer){
  const view=new DataView(buffer),frames=[];let offset=4*LIVE_BATCH;
  if(buffer.byteLength<offset)throw Error('Incomplete webcam response.');
  for(let index=0;index<LIVE_BATCH;index++){
    const size=view.getUint32(4*index),end=offset+size;if(end>buffer.byteLength)throw Error('Incomplete webcam response.');
    frames.push(new Blob([buffer.slice(offset,end)],{type:'image/jpeg'}));offset=end;
  }
  if(offset!==buffer.byteLength)throw Error('Invalid webcam response.');return frames;
}
async function sendLiveBatch(sequence,inputs,cycleStarted,captured){
  const controller=new AbortController();liveRequests.add(controller);
  try{
    const response=await fetch('/webcam/frame?frame='+sequence,{method:'POST',headers:{'X-SAM2-Request':'1'},body:packLiveFrames(inputs),signal:controller.signal}),firstByte=performance.now();
    if(!response.ok){let message=response.statusText;try{message=(await response.json()).error||message;}catch(_){}throw Error(message);}
    const timing=Object.fromEntries([...(response.headers.get('Server-Timing')||'').matchAll(/([a-z0-9-]+);dur=([0-9.]+)/g)].map(match=>[match[1],Number(match[2])]));
    const rendered=unpackLiveFrames(await response.arrayBuffer()),ended=performance.now();
    return {rendered,timing,capture:(captured-cycleStarted)/LIVE_BATCH,netIn:Math.max(0,firstByte-captured-(timing.batch||0)),netOut:ended-firstByte,latency:ended-cycleStarted};
  }finally{liveRequests.delete(controller);}
}
function cancelLiveRequests(){for(const request of liveRequests)request.abort();liveRequests.clear();}
async function enqueueLive(sequence,pending){
  const cycleStarted=performance.now(),inputs=[];
  for(let index=0;index<LIVE_BATCH;index++){await nextCameraFrame();if(!live)return;inputs.push(await cameraFrame());if(!live)return;}
  const captured=performance.now();pending.set(sequence,sendLiveBatch(sequence*LIVE_BATCH,inputs,cycleStarted,captured));
}
async function fillLive(pending,state){while(live&&pending.size<state.depth)await enqueueLive(state.next++,pending);}
function adjustLiveDepth(result,state){
  const timing=result.timing,n300=timing.n300||0,idle=timing['input-wait']||0;
  // Frames within a batch naturally wait behind one another; only count queueing beyond that pipeline fill.
  const allowedIdle=Math.max(2,LIVE_FRAME_MS-n300),excessQueue=Math.max(0,(timing.queue||0)-(LIVE_BATCH-1)*n300/2);
  state.batches++;
  // Grow immediately on starvation, but shrink only after a stable period with excess buffered work.
  if(state.batches>1&&idle>allowedIdle+LIVE_IDLE_SLACK_MS&&state.depth<LIVE_MAX_DEPTH){
    state.depth++;state.queueStreak=0;state.decreaseHold=LIVE_DEPTH_HOLD_BATCHES;return;
  }
  if(state.decreaseHold){state.decreaseHold--;return;}
  state.queueStreak=excessQueue>n300?state.queueStreak+1:0;
  if(state.queueStreak>=LIVE_QUEUE_STREAK&&state.depth>LIVE_MIN_DEPTH){
    state.depth--;state.queueStreak=0;state.decreaseHold=LIVE_QUEUE_STREAK;
  }
}
async function liveLoop(){
  const pending=new Map(),state={next:0,depth:LIVE_START_DEPTH,batches:0,queueStreak:0,decreaseHold:0};let batch=0,display=0,firstDisplayed=null,lastDisplayed=null;
  try{
    status.textContent='Warming up live tracking...';
    await fillLive(pending,state);
    while(pending.size){
      const result=await pending.get(batch);pending.delete(batch++);
      adjustLiveDepth(result,state);const refill=live?fillLive(pending,state):null;
      for(const rendered of result.rendered){
        const delay=lastDisplayed===null?0:LIVE_FRAME_MS-(performance.now()-lastDisplayed);if(delay>0)await new Promise(resolve=>setTimeout(resolve,delay));if(!live)break;
        const now=performance.now();lastDisplayed=now;if(firstDisplayed===null)firstDisplayed=now;display++;const fps=display>1?(display-1)*1000/Math.max(1,now-firstDisplayed):0;
        const url=URL.createObjectURL(rendered);image.src=url;image.hidden=false;video.hidden=true;
        if(liveImageUrl)URL.revokeObjectURL(liveImageUrl);liveImageUrl=url;const timing=result.timing;
        const label=display<LIVE_START_DEPTH*LIVE_BATCH?`Warming up ${display}/${LIVE_START_DEPTH*LIVE_BATCH}`:`Live ${display}`;
        status.textContent=`${label} (${fps.toFixed(1)} FPS) | depth ${state.depth} | capture ${result.capture.toFixed(1)} ms | net->host ~${result.netIn.toFixed(1)} ms | host pre ${(timing['host-pre']||0).toFixed(1)} ms | queue ${(timing.queue||0).toFixed(1)} ms | N300 ${(timing.n300||0).toFixed(1)} ms | idle ${(timing['input-wait']||0).toFixed(1)} ms | host post ${(timing['host-post']||0).toFixed(1)} ms | net->browser ${result.netOut.toFixed(1)} ms | latency ${result.latency.toFixed(1)} ms`;
      }
      if(refill)await refill;
    }
  }catch(error){
    if(!live){await Promise.allSettled(pending.values());return;}
    live=false;cancelLiveRequests();setBusy(true,'Stopping webcam tracking...');const stopping=stopWebcamServer();
    await Promise.allSettled(pending.values());await finishWebcam('Live tracking stopped: '+error.message,stopping);setBusy(false);
  }
}
async function stopLive(){
  live=false;cancelLiveRequests();setBusy(true,'Stopping webcam tracking...');const stopping=stopWebcamServer();
  if(liveTask)await liveTask;liveTask=null;await finishWebcam('Webcam tracking stopped.',stopping);setBusy(false);
}
track.onclick=async()=>{
  if(live){await stopLive();return;}
  setBusy(true,'Starting tracking...');prompt.hidden=true;
  try{
    const result=await api('/track',{method:'POST'});
    if(webcamInput){live=true;status.textContent='Webcam tracking is live.';}
    else{
      const source='/output?v='+Date.now();video.srcObject=null;video.src=source;video.controls=true;video.hidden=false;image.hidden=true;video.load();
      download.href=source;download.download=result.filename;download.hidden=false;finished=true;status.textContent='Tracking complete.';
    }
  }catch(error){prompt.hidden=false;status.textContent=error.message;}
  finally{setBusy(false);}
  if(live)liveTask=liveLoop();
};
let polling=false;
setInterval(async()=>{
  if(!busy||polling)return;polling=true;const generation=operation;
  try{
    const state=await (await fetch('/status',{cache:'no-store'})).json();
    if(busy&&generation===operation){status.textContent=state.message;if(state.total){progress.hidden=false;progress.max=state.total;progress.value=state.current||0;}else progress.hidden=true;}
  }catch(_){}finally{polling=false;}
},400);
video.onerror=()=>{if(!camera)status.textContent='Inline playback failed; use Download result.';};
window.addEventListener('beforeunload',()=>{stopCamera();if(live)fetch('/webcam/stop',{method:'POST',headers:{'X-SAM2-Request':'1'},keepalive:true});});
</script>
"""


def _prompt_inputs(points, box, image_size):
    width, height = image_size
    if len(points) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} points are supported")
    prompts = {}
    if points:
        coordinates, labels = [], []
        for point in points:
            if len(point) != 3 or int(point[2]) not in (0, 1):
                raise ValueError("points must contain x, y, and a 0/1 label")
            x = min(max(float(point[0]), 0), width - 1) * MODEL_IMAGE_SIZE / width
            y = min(max(float(point[1]), 0), height - 1) * MODEL_IMAGE_SIZE / height
            coordinates.append([x, y])
            labels.append(int(point[2]))
        prompts["input_points"] = torch.tensor([[coordinates]], dtype=torch.float32)
        prompts["input_labels"] = torch.tensor([[labels]], dtype=torch.int32)
    if box is not None:
        if len(box) != 4:
            raise ValueError("box must contain x0, y0, x1, and y1")
        x0, y0, x1, y1 = (float(value) for value in box)
        x0, x1 = sorted((min(max(x0, 0), width - 1), min(max(x1, 0), width - 1)))
        y0, y1 = sorted((min(max(y0, 0), height - 1), min(max(y1, 0), height - 1)))
        if x0 == x1 or y0 == y1:
            raise ValueError("box must have non-zero width and height")
        prompts["input_boxes"] = torch.tensor(
            [
                [
                    [
                        x0 * MODEL_IMAGE_SIZE / width,
                        y0 * MODEL_IMAGE_SIZE / height,
                        x1 * MODEL_IMAGE_SIZE / width,
                        y1 * MODEL_IMAGE_SIZE / height,
                    ]
                ]
            ],
            dtype=torch.float32,
        )
    if not prompts:
        raise ValueError("add a point or box prompt first")
    return prompts


def _pixel_values(frame):
    pixels = cv2.dnn.blobFromImage(frame, 1 / 255, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), IMAGE_MEAN_255)
    pixels /= IMAGE_STD
    return torch.from_numpy(pixels)


def _mask_array(mask, image_size):
    width, height = image_size
    mask = torch.as_tensor(mask).float()
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    mask = mask.cpu().numpy()
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got {mask.shape}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask > 0


def _render(frame, mask):
    selected = _mask_array(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8)
    rendered = frame.copy()
    cv2.copyTo(cv2.LUT(frame, MASK_LUT), selected, rendered)
    return rendered


class WebcamPipeline:
    def __init__(self, session):
        self.session = session
        self.condition = threading.Condition()
        self.items = {}
        self.next_sequence = 0
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name="sam2-webcam", daemon=True)
        self.thread.start()

    def submit(self, frames):
        frames = list(frames)
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"webcam pipeline failed: {self.error}") from self.error
            if self.closed:
                raise RuntimeError("webcam pipeline is closed")
            sequences = [sequence for sequence, _ in frames]
            if len(sequences) != len(set(sequences)):
                raise ValueError("duplicate webcam frame")
            for sequence in sequences:
                if sequence < self.next_sequence or sequence in self.items:
                    raise ValueError(f"duplicate or stale webcam frame {sequence}")
                if sequence >= self.next_sequence + LIVE_PIPELINE_DEPTH * LIVE_BATCH_SIZE:
                    raise ValueError(f"webcam frame {sequence} exceeds the bounded reorder window")
            submitted = time.perf_counter()
            items = [{"pixels": pixels, "ready": threading.Event(), "submitted": submitted} for _, pixels in frames]
            self.items.update((sequence, item) for sequence, item in zip(sequences, items))
            self.condition.notify_all()
        results = []
        for item in items:
            item["ready"].wait()
            if "error" in item:
                raise RuntimeError(f"webcam pipeline failed: {item['error']}") from item["error"]
            results.append((item["mask"], item["queue_ms"], item["n300_ms"], item["input_wait_ms"]))
        return results

    def _run(self):
        pending = deque()
        input_wait_ms = 0.0

        def frames():
            nonlocal input_wait_ms
            while True:
                started = time.perf_counter()
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.next_sequence in self.items)
                    input_wait_ms += (time.perf_counter() - started) * 1000
                    item = self.items.pop(self.next_sequence, None)
                    if item is None:
                        return
                    self.next_sequence += 1
                    self.condition.notify_all()
                pending.append(item)
                yield item.pop("pixels")

        try:
            outputs = iter(self.session.run(frames(), None))
            while True:
                wait_before = input_wait_ms
                started = time.perf_counter()
                try:
                    output = next(outputs)
                except StopIteration:
                    break
                item = pending[0]
                mask = ttnn.to_torch(output["pred_masks_high_res"])
                pending.popleft()
                finished = time.perf_counter()
                item["input_wait_ms"] = input_wait_ms - wait_before
                n300_ms = max(0.0, (finished - started) * 1000 - item["input_wait_ms"])
                item["mask"] = mask
                item["n300_ms"] = n300_ms
                item["queue_ms"] = max(0.0, (finished - item["submitted"]) * 1000 - n300_ms)
                item["ready"].set()
        except BaseException as error:
            self.error = error
        finally:
            with self.condition:
                self.closed = True
                abandoned = [*self.items.values(), *pending]
                self.items.clear()
                self.condition.notify_all()
            error = self.error or RuntimeError("webcam pipeline stopped")
            for item in abandoned:
                if not item["ready"].is_set():
                    item["error"] = error
                    item["ready"].set()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()
        session, self.session = self.session, None
        if session is not None:
            session.close()


class Sam2Demo:
    def __init__(self):
        # Avoid host-worker oversubscription starving TT dispatch.
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="sam2-demo-")
        self.directory = Path(self.temporary_directory.name)
        self.operation_lock = threading.Lock()
        self._status = {"message": "Choose an image, video, or webcam.", "current": 0, "total": 0}
        self.path = self.frame = self.preview_path = self.mask_path = self.output_path = None
        self.filename = self.processed = self.tracking_prompts = self.webcam_pipeline = None
        self.is_video = self.is_webcam = self.image_encoded = False
        self.mesh_device = self.model = self.processor = None
        self._closed = False

    def set_status(self, message, current=0, total=0):
        self._status = {"message": message, "current": current, "total": total}

    def status(self):
        return {**self._status, "busy": self.operation_lock.locked()}

    def upload(self, name, source, size, *, webcam=False):
        filename = Path(name).name
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS or webcam and suffix not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported file type: {suffix}")
        if not 0 < size <= MAX_UPLOAD_BYTES:
            raise ValueError("upload must be between 1 byte and 2 GiB")
        self.stop_webcam()
        if self.model is not None:
            self.model.reset_image()
        for old_path in self.directory.iterdir():
            old_path.unlink()
        self.path = self.frame = self.preview_path = self.mask_path = self.output_path = None
        self.processed = self.tracking_prompts = None
        self.is_video = self.is_webcam = self.image_encoded = False
        path = self.directory / f"input{suffix}"
        remaining, total = size, size
        try:
            with path.open("wb") as destination:
                while remaining:
                    chunk = source.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("incomplete upload")
                    destination.write(chunk)
                    remaining -= len(chunk)
                    self.set_status("Uploading media...", total - remaining, total)
            self.set_status("Decoding the first frame...")
            if suffix in IMAGE_EXTENSIONS:
                with Image.open(path) as image:
                    frame = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
                is_video = webcam
            else:
                capture = cv2.VideoCapture(str(path))
                ok, bgr = capture.read()
                capture.release()
                if not ok:
                    raise ValueError(f"cannot read {filename}")
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                is_video = True
        except Exception:
            path.unlink(missing_ok=True)
            raise

        self.preview_path = self.directory / "preview.jpg"
        Image.fromarray(frame).save(self.preview_path, quality=90)
        self.path, self.frame, self.filename, self.is_video, self.is_webcam = path, frame, filename, is_video, webcam
        height, width = frame.shape[:2]
        self.set_status("Ready for a prompt.")
        return {"video": is_video, "webcam": webcam, "width": width, "height": height}

    def ensure_model(self):
        if self.model is not None:
            return
        self.set_status("Opening the N300 mesh...")
        try:
            self.mesh_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=SAM2_L1_SMALL_SIZE, num_command_queues=2
            )
            self.set_status("Loading the SAM2 checkpoint and processor...")
            hf_model, self.processor = load_sam2_model_and_processor()
            self.set_status("Building the TTNN model...")
            self.model = build_tt_sam2_model(hf_model, self.mesh_device, bridge_upload_cq_id=1)
            self.set_status("Choose an image, video, or webcam.")
        except BaseException:
            self.close_model()
            raise

    def clear_prompts(self):
        self.stop_webcam()
        self.tracking_prompts = self.mask_path = self.output_path = None
        self.set_status("Prompts cleared.")
        return {}

    def segment(self, points, box):
        if self.path is None:
            raise ValueError("upload an image/video or capture a webcam frame first")
        height, width = self.frame.shape[:2]
        prompts = _prompt_inputs(points, box, (width, height))
        self.ensure_model()
        if self.processed is None:
            self.set_status("Preparing the first frame...")
            self.processed = self.processor(images=Image.fromarray(self.frame), return_tensors="pt")
        if not self.image_encoded:
            self.set_status("Encoding the first frame on N300...")
            self.model.set_image(self.processed.pixel_values)
            self.image_encoded = True
        self.set_status(f"Generating mask from {len(points)} point(s){' and a box' if box else ''}...")
        output = None
        try:
            output = self.model.predict(**prompts, multimask_output=True)
            masks = ttnn.to_torch(output["low_res_masks"]).float()
            scores = ttnn.to_torch(output["iou_scores"]).float()
            best = int(torch.argmax(scores, dim=-1).item())
            mask = self.processor.post_process_masks(
                [masks[:, best : best + 1]], self.processed.original_sizes, binarize=True
            )[0]
            self.set_status("Rendering the selected mask...")
            self.mask_path = self.directory / f"{Path(self.filename).stem}_sam2.png"
            Image.fromarray(_render(self.frame, mask)).save(self.mask_path)
            self.tracking_prompts = prompts
            self.output_path = None if self.is_video else self.mask_path
            self.set_status("Mask ready.")
            return {"filename": self.mask_path.name, "video": self.is_video}
        finally:
            if output is not None:
                for tensor in output.values():
                    ttnn.deallocate(tensor)

    def track(self):
        if not self.is_video:
            raise ValueError("tracking requires a video or webcam")
        if self.tracking_prompts is None or self.mask_path is None:
            raise ValueError("generate and review the first-frame mask before tracking")
        self.image_encoded = False
        if self.is_webcam:
            self.set_status("Starting webcam tracking...")
            session = self.model.start_video_session()
            try:
                session.step(self.processed.pixel_values, self.tracking_prompts)
                self.webcam_pipeline = WebcamPipeline(session)
            except BaseException:
                session.close()
                raise
            self.set_status("Webcam tracking is live.")
            return {"live": True}
        self.output_path = self._run_video(self.tracking_prompts)
        self.set_status("Tracking complete.")
        return {"filename": self.output_path.name, "video": True}

    def webcam_frames(self, encoded, sequence):
        pipeline = self.webcam_pipeline
        if pipeline is None:
            raise ValueError("start webcam tracking first")
        batch_started = time.perf_counter()
        header_size = 4 * LIVE_BATCH_SIZE
        if len(encoded) < header_size:
            raise ValueError("incomplete webcam batch")
        sizes = [int.from_bytes(encoded[index : index + 4], "big") for index in range(0, header_size, 4)]
        offset, prepared = header_size, []
        for size in sizes:
            if not 0 < size <= MAX_WEBCAM_FRAME_BYTES or offset + size > len(encoded):
                raise ValueError("invalid webcam frame size")
            started = time.perf_counter()
            bgr = cv2.imdecode(np.frombuffer(encoded[offset : offset + size], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cannot decode webcam frame")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prepared.append((frame, _pixel_values(frame), (time.perf_counter() - started) * 1000))
            offset += size
        if offset != len(encoded):
            raise ValueError("invalid webcam batch")

        outputs = pipeline.submit((sequence + index, item[1]) for index, item in enumerate(prepared))
        results, samples = [], []
        for (frame, _, host_pre_ms), (mask, queue_ms, n300_ms, input_wait_ms) in zip(prepared, outputs):
            started = time.perf_counter()
            rendered = _render(frame, mask)
            ok, result = cv2.imencode(".jpg", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("cannot encode tracked webcam frame")
            samples.append((host_pre_ms, queue_ms, n300_ms, input_wait_ms, (time.perf_counter() - started) * 1000))
            results.append(result.tobytes())
        names = ("host-pre", "queue", "n300", "input-wait", "host-post")
        timings = tuple(
            (name, sum(sample[index] for sample in samples) / len(samples)) for index, name in enumerate(names)
        )
        timings += (("batch", (time.perf_counter() - batch_started) * 1000),)
        body = b"".join(len(result).to_bytes(4, "big") for result in results) + b"".join(results)
        return body, timings

    def stop_webcam(self):
        pipeline, self.webcam_pipeline = self.webcam_pipeline, None
        if pipeline is not None:
            pipeline.close()
        return {}

    def _run_video(self, prompts):
        output_path = self.directory / f"{Path(self.filename).stem}_sam2.mp4"
        with ExitStack() as stack:
            capture = cv2.VideoCapture(str(self.path))
            stack.callback(capture.release)
            started = time.perf_counter()
            ok, first_bgr = capture.read()
            first_decode_ms = (time.perf_counter() - started) * 1000
            if not ok:
                raise ValueError(f"cannot read {self.filename}")
            height, width = first_bgr.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            container = stack.enter_context(av.open(str(output_path), mode="w"))
            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "23", "threads": "4"}

            def flush_writer():
                for packet in stream.encode():
                    container.mux(packet)

            stack.callback(flush_writer)
            pending_frames, pending_pre_ms = deque(), deque()
            self.set_status("Capturing the video encoder trace...")
            session = self.model.start_video_session()
            stack.callback(session.close)
            pre_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam2-video-in"))
            post_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam2-video-out"))

            def prepare_frame(bgr, decode_ms):
                started = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pixels = _pixel_values(rgb)
                return rgb, pixels, decode_ms + (time.perf_counter() - started) * 1000

            def read_frame():
                started = time.perf_counter()
                ok, bgr = capture.read()
                decode_ms = (time.perf_counter() - started) * 1000
                return prepare_frame(bgr, decode_ms) if ok else None

            pre_future = pre_executor.submit(prepare_frame, first_bgr, first_decode_ms)

            def pixel_frames():
                nonlocal pre_future
                while pre_future is not None:
                    prepared = pre_future.result()
                    if prepared is None:
                        pre_future = None
                        return
                    pre_future = pre_executor.submit(read_frame)
                    rgb, pixels, pre_ms = prepared
                    pending_frames.append(rgb)
                    pending_pre_ms.append(pre_ms)
                    yield pixels

            def render_frame(frame, mask):
                started = time.perf_counter()
                rendered = _render(frame, mask)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rendered, format="rgb24")):
                    container.mux(packet)
                return (time.perf_counter() - started) * 1000

            frames = pixel_frames()
            stack.callback(frames.close)
            outputs = iter(session.run(frames, prompts))
            frame_count = 0
            timing_window = deque(maxlen=16)
            post_futures = deque()

            def report_timing(frame_index, pre_ms, n300_ms, post_ms):
                sample = (pre_ms, n300_ms, post_ms, max(pre_ms, n300_ms, post_ms))
                if frame_index == 1:
                    averages = sample
                else:
                    timing_window.append(sample)
                    averages = tuple(sum(values) / len(timing_window) for values in zip(*timing_window))
                pre_avg, n300_avg, post_avg, frame_avg = averages
                frame_label = f"{frame_index}{f'/{total}' if total else ''}"
                self.set_status(
                    f"Video {frame_label} | host pre {pre_avg:.1f} ms | N300 pipeline {n300_avg:.1f} ms | "
                    f"host post {post_avg:.1f} ms | throughput {frame_avg:.1f} ms ({1000 / frame_avg:.1f} FPS)",
                    frame_index,
                    total,
                )

            while True:
                started = time.perf_counter()
                try:
                    output = next(outputs)
                except StopIteration:
                    break
                mask = ttnn.to_torch(output["pred_masks_high_res"])
                n300_ms = (time.perf_counter() - started) * 1000
                frame_count += 1
                post_futures.append(
                    (
                        (frame_count, pending_pre_ms.popleft(), n300_ms),
                        post_executor.submit(render_frame, pending_frames.popleft(), mask),
                    )
                )
                if len(post_futures) >= VIDEO_POST_DEPTH:
                    pending_timing, post_future = post_futures.popleft()
                    report_timing(*pending_timing, post_future.result())
            while post_futures:
                pending_timing, post_future = post_futures.popleft()
                report_timing(*pending_timing, post_future.result())
            if not frame_count:
                raise ValueError(f"{self.filename} contains no frames")
        return output_path

    def close_model(self):
        self.stop_webcam()
        model, self.model = self.model, None
        mesh_device, self.mesh_device = self.mesh_device, None
        try:
            if model is not None:
                model.close()
        finally:
            if mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.close_model()
        finally:
            self.temporary_directory.cleanup()


class DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    demo = None

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body, content_type, timings=()):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if timings:
            self.send_header("Server-Timing", ", ".join(f"{name};dur={duration:.3f}" for name, duration in timings))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, allow_range=False):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested_range = self.headers.get("Range") if allow_range else None
        if requested_range:
            try:
                unit, value = requested_range.split("=", 1)
                first, last = value.split("-", 1)
                if unit != "bytes" or "," in value:
                    raise ValueError
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    raise ValueError
                status = 206
            except (TypeError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        if allow_range:
            self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _body(self, max_size):
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= max_size:
            raise ValueError("invalid request body size")
        body = self.rfile.read(size)
        if len(body) != size:
            raise ValueError("incomplete request body")
        return body

    def _request(self, *, required=True):
        if not required and int(self.headers.get("Content-Length", 0)) == 0:
            return {}
        return json.loads(self._body(64 * 1024))

    def do_GET(self):
        target = urlparse(self.path)
        try:
            if target.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif target.path == "/status":
                self._json(self.demo.status())
            elif target.path == "/preview" and self.demo.preview_path is not None:
                self._file(self.demo.preview_path)
            elif target.path == "/mask" and self.demo.mask_path is not None:
                self._file(self.demo.mask_path)
            elif target.path == "/output" and self.demo.output_path is not None:
                self._file(self.demo.output_path, allow_range=True)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        target = urlparse(self.path)
        if self.headers.get("X-SAM2-Request") != "1":
            self.close_connection = True
            self._json({"error": "forbidden"}, 403)
            return
        if target.path == "/webcam/frame":
            try:
                sequence = int(parse_qs(target.query).get("frame", [None])[0])
                body, timings = self.demo.webcam_frames(
                    self._body(LIVE_BATCH_SIZE * (MAX_WEBCAM_FRAME_BYTES + 4)), sequence
                )
                self._bytes(body, "application/octet-stream", timings)
            except ConnectionError:
                pass
            except Exception as error:
                self.close_connection = True
                self.demo.set_status(f"Error: {error}")
                self._json({"error": str(error)}, 500)
            return
        if not self.demo.operation_lock.acquire(blocking=False):
            self.close_connection = True
            self._json({"error": "another operation is still running"}, 409)
            return
        try:
            if target.path in ("/segment", "/track") and self.demo.webcam_pipeline is not None:
                self.close_connection = True
                self._json({"error": "stop webcam tracking before starting another operation"}, 409)
                return
            if target.path == "/upload":
                query = parse_qs(target.query)
                name = query.get("name", [None])[0]
                if not name:
                    raise ValueError("missing filename")
                result = self.demo.upload(
                    name,
                    self.rfile,
                    int(self.headers.get("Content-Length", 0)),
                    webcam=query.get("webcam") == ["1"],
                )
            elif target.path == "/segment":
                request = self._request()
                result = self.demo.segment(request.get("points", []), request.get("box"))
            elif target.path == "/clear":
                self._request(required=False)
                result = self.demo.clear_prompts()
            elif target.path == "/track":
                self._request(required=False)
                result = self.demo.track()
            elif target.path == "/webcam/stop":
                self._request(required=False)
                result = self.demo.stop_webcam()
            else:
                self.close_connection = True
                self._json({"error": "not found"}, 404)
                return
            self._json(result)
        except Exception as error:
            self.close_connection = True
            self.demo.set_status(f"Error: {error}")
            self._json({"error": str(error)}, 500)
        finally:
            self.demo.operation_lock.release()

    def log_message(self, format, *args):
        if urlparse(self.path).path != "/status":
            super().log_message(format, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    demo = Sam2Demo()
    DemoHandler.demo = demo
    try:
        demo.ensure_model()
        server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
        print(f"SAM2 demo: http://127.0.0.1:{args.port}")
        shutdown_started = threading.Event()

        def stop_server(_signal_number, _frame):
            if not shutdown_started.is_set():
                shutdown_started.set()
                threading.Thread(target=server.shutdown, name="sam2-server-shutdown", daemon=True).start()

        for signal_number in (signal.SIGHUP, signal.SIGTERM):
            signal.signal(signal_number, stop_server)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            demo.stop_webcam()
            server.server_close()
    finally:
        with demo.operation_lock:
            demo.close()
