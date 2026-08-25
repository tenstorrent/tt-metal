# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-language sentences for the tests that drive every supported language.

Three per language, of growing length, so a caller can take one or use all three. Keyed by the
codes in frontend.SUPPORTED_LANGUAGES; the tests that use this assert the keys still match, so
adding a language without text fails rather than silently going untested.

Shared because two tests need it and neither owns it: test_all_languages_smoke drives all three
lengths through the device, and test_model_teacher_forced_pcc takes the first of each to compare
latents against the CPU.
"""
SENTENCES = {
    "en": [
        "The winter market opened early today.",
        "She folded the letter and left it on the table.",
        "Migrating birds crossed the valley at dusk.",
    ],
    "de": [
        "Der Wintermarkt öffnete heute früh.",
        "Sie faltete den Brief und ließ ihn auf dem Tisch.",
        "Zugvögel überquerten das Tal in der Abenddämmerung.",
    ],
    "fr": [
        "Le marché d'hiver a ouvert tôt aujourd'hui.",
        "Elle a plié la lettre et l'a laissée sur la table.",
        "Les oiseaux migrateurs ont traversé la vallée au crépuscule.",
    ],
    "es": [
        "El mercado de invierno abrió temprano hoy.",
        "Ella dobló la carta y la dejó sobre la mesa.",
        "Las aves migratorias cruzaron el valle al atardecer.",
    ],
    "it": [
        "Il mercato invernale ha aperto presto oggi.",
        "Ha piegato la lettera e l'ha lasciata sul tavolo.",
        "Gli uccelli migratori hanno attraversato la valle al crepuscolo.",
    ],
    "pt": [
        "O mercado de inverno abriu cedo hoje.",
        "Ela dobrou a carta e deixou-a na mesa.",
        "As aves migratórias cruzaram o vale ao anoitecer.",
    ],
    "nl": [
        "De wintermarkt opende vandaag vroeg.",
        "Ze vouwde de brief en liet hem op de tafel liggen.",
        "Trekvogels staken bij schemering de vallei over.",
    ],
    "pl": [
        "Zimowy targ otworzył się dziś wcześnie.",
        "Złożyła list i zostawiła go na stole.",
        "Ptaki wędrowne przeleciały nad doliną o zmierzchu.",
    ],
    "tr": [
        "Kış pazarı bugün erken açıldı.",
        "Mektubu katladı ve masaya bıraktı.",
        "Göçmen kuşlar akşam vakti vadiyi geçti.",
    ],
    "ru": [
        "Зимний рынок открылся сегодня рано.",
        "Она сложила письмо и оставила его на столе.",
        "Перелётные птицы пересекли долину в сумерках.",
    ],
    "hu": [
        "A téli piac ma korán kinyitott.",
        "Összehajtotta a levelet és az asztalon hagyta.",
        "A vándormadarak alkonyatkor átkeltek a völgyön.",
    ],
    "ar": [
        "افتتح سوق الشتاء مبكرا اليوم.",
        "طوت الرسالة وتركتها على الطاولة.",
        "عبرت الطيور المهاجرة الوادي عند الغروب.",
    ],
    "hi": [
        "सर्दियों का बाज़ार आज जल्दी खुला।",
        "उसने पत्र मोड़ा और मेज़ पर छोड़ दिया।",
        "प्रवासी पक्षी शाम के समय घाटी पार कर गए।",
    ],
    "ko": [
        "겨울 시장이 오늘 일찍 열렸다.",
        "그녀는 편지를 접어 탁자 위에 두었다.",
        "철새들이 해질 무렵에 계곡을 건넜다.",
    ],
    "zh": [
        "冬季市场今天很早就开了。",
        "她把信折好放在桌子上。",
        "候鸟在黄昏时飞过山谷。",
    ],
    "ja": [
        "冬の市場は今日早く開いた。",
        "彼女は手紙をたたんで机の上に置いた。",
        "渡り鳥が夕暮れに谷を越えていった。",
    ],
}


# --- WER corpus -----------------------------------------------------------------------------
# Long sentences: short utterances swing too widely between seeds to gate on. No numbers in either
# form -- the cleaner expands digits, and the ASR writes spelled-out ones back as numerals. Nothing
# the ASR respells; ordinary prose. Arabic is undiacritized, matching what the recogniser emits.
WER_SENTENCES = {
    "en": [
        "The old map showed three islands that no sailor had ever found, and nobody wanted to be "
        "the first to erase them.",
        "Warm bread and strong coffee filled the small kitchen long before the sun came up, and the "
        "kettle went on again the moment the first pot stood empty.",
        "The archive filled four floors of a building designed for something else entirely, and "
        "every year the collection pressed a little harder against the walls until the shelves "
        "reached the ceiling.",
        "The bakery on the corner opens before dawn, and by six the whole street smells of warm "
        "bread, which is why the regulars arrive in the same order every morning without ever "
        "arranging it.",
        "Long before the railway reached the valley, the mail came over the mountain pass on "
        "horseback, and the timetable was a matter of weather rather than clocks, so the villagers "
        "learned to read the sky the way other people read a schedule.",
    ],
    "de": [
        "Der alte Stadtplan zeigte drei Inseln, die kein Seemann je gefunden hatte, und niemand "
        "wollte der erste sein, der sie wieder auslöschte.",
        "Warmes Brot und starker Kaffee füllten die kleine Küche lange bevor die Sonne aufging, und "
        "der Kessel kam wieder auf den Herd, sobald die erste Kanne leer stand.",
        "Das Archiv füllte vier Stockwerke eines Gebäudes, das für etwas völlig anderes entworfen "
        "worden war, und jedes Jahr drückte die Sammlung ein wenig härter gegen die Wände.",
        "Die Bäckerei an der Ecke öffnet vor dem Morgengrauen, und die ganze Straße riecht dann nach "
        "warmem Brot, weshalb die Stammgäste jeden Morgen in derselben Reihenfolge erscheinen.",
        "Lange bevor die Eisenbahn das Tal erreichte, kam die Post zu Pferd über den Bergpass, und "
        "der Fahrplan war eine Frage des Wetters und nicht der Uhren, sodass die Dorfbewohner "
        "lernten, den Himmel zu lesen.",
    ],
    "fr": [
        "La vieille carte montrait trois îles qu'aucun marin n'avait jamais trouvées, et personne ne "
        "voulait être le premier à les effacer.",
        "Le pain chaud et le café fort remplissaient la petite cuisine bien avant le lever du "
        "soleil, et la bouilloire repartait dès que la première cafetière restait vide.",
        "Les archives occupaient quatre étages d'un bâtiment conçu pour tout autre chose, et chaque "
        "année la collection pressait un peu plus fort contre les murs.",
        "La boulangerie du coin ouvre avant l'aube, et toute la rue sent alors le pain chaud, ce qui "
        "explique pourquoi les habitués arrivent dans le même ordre chaque matin.",
        "Bien avant que le chemin de fer n'atteigne la vallée, le courrier passait le col à cheval, "
        "et l'horaire dépendait du temps plutôt que des horloges, si bien que les villageois "
        "apprirent à lire le ciel.",
    ],
    "es": [
        "El mapa antiguo mostraba tres islas que ningún marinero había encontrado nunca, y nadie "
        "quería ser el primero en borrarlas.",
        "El pan caliente y el café fuerte llenaban la pequeña cocina mucho antes de que saliera el "
        "sol, y la tetera volvía al fuego en cuanto la primera jarra quedaba vacía.",
        "El archivo ocupaba cuatro plantas de un edificio diseñado para algo completamente distinto, "
        "y cada año la colección apretaba un poco más contra las paredes.",
        "La panadería de la esquina abre antes del amanecer, y toda la calle huele entonces a pan "
        "caliente, por lo que los clientes habituales llegan en el mismo orden cada mañana.",
        "Mucho antes de que el ferrocarril llegara al valle, el correo cruzaba el puerto de montaña "
        "a caballo, y el horario era una cuestión del tiempo y no de los relojes, así que los "
        "aldeanos aprendieron a leer el cielo.",
    ],
    "it": [
        "La vecchia mappa mostrava tre isole che nessun marinaio aveva mai trovato, e nessuno voleva "
        "essere il primo a cancellarle.",
        "Il pane caldo e il caffè forte riempivano la piccola cucina molto prima che il sole si "
        "alzasse, e il bollitore tornava sul fuoco appena la prima caffettiera restava vuota.",
        "L'archivio occupava quattro piani di un edificio progettato per qualcosa di completamente "
        "diverso, e ogni anno la collezione premeva un poco più forte contro le pareti.",
        "La panetteria all'angolo apre prima dell'alba, e tutta la strada profuma allora di pane "
        "caldo, ed è per questo che i clienti abituali arrivano nello stesso ordine ogni mattina.",
        "Molto prima che la ferrovia raggiungesse la valle, la posta passava il colle a cavallo, e "
        "l'orario era una questione di tempo e non di orologi, così gli abitanti impararono a "
        "leggere il cielo.",
    ],
    "pt": [
        "O mapa antigo mostrava três ilhas que nenhum marinheiro jamais encontrara, e ninguém "
        "queria ser o primeiro a apagá-las.",
        "O pão quente e o café forte enchiam a pequena cozinha muito antes de o sol nascer, e a "
        "chaleira voltava ao fogo assim que o primeiro bule ficava vazio.",
        "O arquivo ocupava quatro andares de um edifício projetado para algo completamente "
        "diferente, e cada ano a coleção pressionava um pouco mais contra as paredes.",
        "A padaria da esquina abre antes do amanhecer, e toda a rua fica então com cheiro de pão "
        "quente, e é por isso que os fregueses chegam na mesma ordem todas as manhãs.",
        "Muito antes de a ferrovia chegar ao vale, o correio atravessava a serra a cavalo, e o "
        "horário era uma questão de tempo e não de relógios, de modo que os moradores aprenderam a "
        "ler o céu.",
    ],
    "nl": [
        "De oude kaart toonde drie eilanden die geen enkele zeeman ooit had gevonden, en niemand "
        "wilde de eerste zijn die ze uitwiste.",
        "Warm brood en sterke koffie vulden de kleine keuken lang voordat de zon opkwam, en de ketel "
        "ging weer op het vuur zodra de eerste kan leeg stond.",
        "Het archief vulde vier verdiepingen van een gebouw dat voor iets heel anders was ontworpen, "
        "en elk jaar drukte de collectie een beetje harder tegen de muren.",
        "De bakkerij op de hoek opent voor zonsopgang, en de hele straat ruikt dan naar warm brood, "
        "en daarom komen de vaste klanten elke ochtend in dezelfde volgorde.",
        "Lang voordat de spoorlijn de vallei bereikte, kwam de post te paard over de bergpas, en de "
        "tijdtabel was een kwestie van het weer en niet van klokken, zodat de dorpelingen leerden "
        "de lucht te lezen.",
    ],
    "pl": [
        "Stara mapa pokazywała trzy wyspy, których żaden żeglarz nigdy nie znalazł, i nikt nie "
        "chciał być pierwszym, który je wymaże.",
        "Ciepły chleb i mocna kawa wypełniały małą kuchnię długo przed wschodem słońca, a czajnik "
        "wracał na ogień, gdy tylko pierwszy dzbanek stał pusty.",
        "Archiwum zajmowało cztery piętra budynku zaprojektowanego do czegoś zupełnie innego, a "
        "każdego roku zbiory naciskały trochę mocniej na ściany.",
        "Piekarnia na rogu otwiera się przed świtem, a cała ulica pachnie wtedy ciepłym chlebem, "
        "dlatego stali klienci przychodzą każdego ranka w tej samej kolejności.",
        "Długo przed tym, jak kolej dotarła do doliny, poczta przechodziła przez przełęcz na koniu, "
        "a rozkład zależał od pogody, a nie od zegarów, więc mieszkańcy nauczyli się czytać niebo.",
    ],
    "tr": [
        "Eski harita, hiçbir denizcinin bulamadığı üç ada gösteriyordu ve kimse onları ilk silen "
        "kişi olmak istemiyordu.",
        "Sıcak ekmek ve sert kahve, güneş doğmadan çok önce küçük mutfağı dolduruyordu ve ilk "
        "demlik boşaldığı anda çaydanlık yeniden ocağa gidiyordu.",
        "Arşiv, tamamen başka bir şey için tasarlanmış bir binanın dört katını doldurmuştu ve her "
        "yıl koleksiyon duvarlara biraz daha sert bastırıyordu.",
        "Köşedeki fırın şafaktan önce açılır ve bütün sokak o saatte sıcak ekmek kokar, bu yüzden "
        "müdavimler her sabah aynı sırayla gelir.",
        "Demiryolu vadiye ulaşmadan çok önce, posta dağ geçidini at üstünde aşıyordu ve tarife "
        "saatlerden çok havaya bağlıydı, bu yüzden köylüler göğü okumayı öğrendiler.",
    ],
    "ru": [
        "Старая карта показывала три острова, которых никогда не находил ни один моряк, и никто не "
        "хотел стать первым, кто их сотрёт.",
        "Тёплый хлеб и крепкий кофе наполняли маленькую кухню задолго до восхода солнца, и чайник "
        "снова ставили на огонь, едва первый кофейник оставался пустым.",
        "Архив занимал четыре этажа здания, спроектированного совсем для другого, и каждый год "
        "собрание давило на стены всё сильнее.",
        "Булочная на углу открывается до рассвета, и вся улица пахнет тогда тёплым хлебом, поэтому "
        "постоянные посетители приходят каждое утро в одном и том же порядке.",
        "Задолго до того, как железная дорога дошла до долины, почту везли через горный перевал на "
        "лошадях, и расписание зависело от погоды, а не от часов, поэтому жители научились читать "
        "небо.",
    ],
    "hu": [
        "A régi térkép három szigetet mutatott, amelyeket egyetlen tengerész sem talált meg soha, "
        "és senki sem akart az első lenni, aki letörli őket.",
        "A meleg kenyér és az erős kávé jóval napkelte előtt megtöltötte a kis konyhát, és a kanna "
        "visszakerült a tűzre, amint az első kancsó kiürült.",
        "A levéltár egy teljesen másra tervezett épület négy emeletét foglalta el, és a gyűjtemény "
        "minden évben egy kicsit erősebben nyomta a falakat.",
        "A sarki pékség hajnal előtt kinyit, és az egész utca meleg kenyértől illatozik, ezért a "
        "törzsvásárlók minden reggel ugyanabban a sorrendben érkeznek.",
        "Jóval azelőtt, hogy a vasút elérte a völgyet, a postát lóháton vitték át a hegyi hágón, és "
        "a menetrend inkább az időjárástól függött, mint az óráktól, ezért a falusiak megtanultak "
        "olvasni az égből.",
    ],
    "ar": [
        "أظهرت الخريطة القديمة ثلاث جزر لم يجدها أي بحار من قبل، ولم يرغب أحد في أن يكون أول من يمحوها.",
        "كان الخبز الساخن والقهوة القوية يملأان المطبخ الصغير قبل شروق الشمس بوقت طويل، وكانت "
        "الغلاية تعود إلى النار بمجرد أن يفرغ الإبريق الأول.",
        "شغل الأرشيف أربعة طوابق من مبنى صمم لشيء آخر تماما، وكانت المجموعة تضغط على الجدران بقوة "
        "أكبر قليلا كل عام.",
        "يفتح المخبز في الزاوية قبل الفجر، ويفوح الشارع كله برائحة الخبز الساخن، ولهذا يأتي الزبائن "
        "المعتادون في الترتيب نفسه كل صباح.",
        "قبل أن يصل القطار إلى الوادي بوقت طويل، كان البريد يعبر الممر الجبلي على ظهر حصان، وكان "
        "الجدول يعتمد على الطقس لا على الساعات، فتعلم القرويون قراءة السماء.",
    ],
    "hi": [
        "पुराने नक्शे में तीन ऐसे द्वीप दिखते थे जिन्हें कोई नाविक कभी नहीं खोज पाया, और कोई भी उन्हें सबसे पहले "
        "मिटाने वाला नहीं बनना चाहता था।",
        "गरम रोटी और तेज़ चाय सूरज निकलने से बहुत पहले ही छोटी रसोई को भर देती थी, और पहला बरतन खाली होते ही "
        "केतली फिर आँच पर चढ़ जाती थी।",
        "वह संग्रह ऐसी इमारत की चार मंज़िलों में फैला था जो किसी और काम के लिए बनी थी, और हर साल संग्रह दीवारों पर "
        "थोड़ा और ज़ोर डालता जाता था।",
        "नुक्कड़ की दुकान भोर से पहले खुल जाती है, और पूरी गली में गरम रोटी की महक फैल जाती है, इसलिए पुराने "
        "ग्राहक हर सुबह उसी क्रम में आते हैं।",
        "रेल के घाटी तक पहुँचने से बहुत पहले, डाक घोड़े पर पहाड़ी दर्रे से आती थी, और समय सारणी घड़ियों से कम और "
        "मौसम से ज़्यादा तय होती थी, इसलिए गाँव वालों ने आसमान पढ़ना सीख लिया।",
    ],
    "ko": [
        "낡은 지도에는 어떤 선원도 찾지 못한 세 개의 섬이 그려져 있었고, 아무도 그것을 먼저 지우고 싶어 하지 않았다.",
        "따뜻한 빵과 진한 커피가 해가 뜨기 훨씬 전부터 작은 부엌을 가득 채웠고, 첫 주전자가 비자마자 물이 다시 " "끓기 시작했다.",
        "그 기록 보관소는 전혀 다른 용도로 지어진 건물의 위층 전체를 차지했고, 해마다 자료가 벽을 조금씩 더 세게 " "밀어냈다.",
        "모퉁이의 빵집은 동트기 전에 문을 열고 거리 전체에 따뜻한 빵 냄새가 퍼지기 때문에 단골들은 매일 아침 " "같은 순서로 찾아온다.",
        "철도가 계곡에 닿기 훨씬 전에는 우편물이 말을 타고 산길을 넘어왔고, 시간표는 시계보다 날씨에 달려 " "있었기에 마을 사람들은 하늘을 읽는 법을 배웠다.",
    ],
    "zh": [
        "旧地图上画着几座从来没有水手找到过的岛屿，也没有人愿意先把它们抹去。",
        "温热的面包和浓咖啡在太阳升起很久以前就填满了那间小厨房，水壶刚空下来就又烧了起来。",
        "那个档案馆占据了一栋为别的用途建造的楼房的上层，每年资料都把墙壁挤得更紧一些。",
        "街角的面包店天亮以前就开门，整条街都飘着温热面包的香气，所以老主顾每天早晨都按同样的顺序前来。",
        "在铁路修到山谷以前很久，邮件由人骑马翻过山口送来，时刻表取决于天气而不是钟表，于是村民学会了看天。",
    ],
    "ja": [
        "古い地図には、どの船乗りも見つけたことのない島がいくつか描かれていて、誰もそれを最初に消したいとは思わなかった。",
        "温かいパンと濃いコーヒーが日の出のずっと前から小さな台所を満たし、やかんは空になるとすぐにまた火にかけられた。",
        "その書庫はまったく別の目的で建てられた建物の上の階を占めていて、毎年資料が壁をいっそう強く押していた。",
        "角のパン屋は夜明け前に店を開け、通り全体に温かいパンの香りが広がるので、常連客は毎朝同じ順番でやって来る。",
        "鉄道が谷に届くよりずっと前、郵便は馬で峠を越えて運ばれ、時刻表は時計よりも天気に左右されたので、村人たちは空を読むことを覚えた。",
    ],
}
